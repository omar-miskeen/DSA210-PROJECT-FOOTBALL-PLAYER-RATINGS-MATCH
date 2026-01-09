import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, levene
import warnings
import os

warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class BarcelonaHypothesisTesting:
    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
            
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Metric Definitions
        self.attacking_metrics = ['npxg', 'shots'] 
        self.creative_metrics = [
            'xg_assist', 
            'sca', 
            'progressive_passes', 
            'take_ons_won',
            'progressive_carries'
        ]
        
        print(f"Initialized analysis. Data shape: {self.df.shape}")
        print(f"Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
    
    def check_assumptions(self):
        print("\n--- Assumption Checks ---")
        
        # 1. Normality Test (Shapiro-Wilk)
        data_rating = self.df['Rating'].dropna()
        if len(data_rating) > 3:
            stat, p = shapiro(data_rating)
            result = "Normal" if p > 0.05 else "Non-normal"
            print(f"Rating Normality: W={stat:.4f}, p={p:.4f} ({result})")

        # 2. Homogeneity of Variance (Levene's Test)
        groups = []
        for r in ['W', 'D', 'L']:
            subset = self.df[self.df['result'].str.contains(r, na=False)]['Rating'].dropna()
            groups.append(subset)
        
        if all(len(g) > 0 for g in groups):
            stat, p = levene(*groups)
            result = "Equal Variances" if p > 0.05 else "Unequal Variances"
            print(f"Variance Homogeneity (W/D/L): W={stat:.4f}, p={p:.4f} ({result})")
    
    def phase1_metrics_to_rating(self):
        print("\n--- Phase 1: Metric vs Rating Correlations ---")
        
        results = []
        all_metrics = self.attacking_metrics + self.creative_metrics
        
        for player in self.df['Player'].unique():
            player_data = self.df[self.df['Player'] == player]
            print(f"Processing {player}...")
            
            for metric in all_metrics:
                if metric in player_data.columns:
                    mask = player_data[metric].notna() & player_data['Rating'].notna()
                    
                    if mask.sum() > 5:
                        corr, p = stats.pearsonr(player_data.loc[mask, metric], player_data.loc[mask, 'Rating'])
                        
                        results.append({
                            'Player': player,
                            'Metric': metric,
                            'Type': 'Attacking' if metric in self.attacking_metrics else 'Creative',
                            'Correlation': corr,
                            'Significant': p < 0.05
                        })
                        
                        sig_marker = "*" if p < 0.05 else ""
                        print(f"  {metric}: r={corr:.3f} {sig_marker}")
        
        self.plot_phase1_results(results)
        return pd.DataFrame(results)
    
    def plot_phase1_results(self, results):
        if not results: return
        df_results = pd.DataFrame(results)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Metric Correlations with Player Rating', fontsize=16)
        
        players = df_results['Player'].unique()
        for idx, player in enumerate(players):
            if idx >= 4: break
            ax = axes[idx // 2, idx % 2]
            p_data = df_results[df_results['Player'] == player].sort_values('Correlation')
            
            colors = ['#ff9999' if t == 'Attacking' else '#66b3ff' for t in p_data['Type']]
            bars = ax.barh(p_data['Metric'], p_data['Correlation'], color=colors, edgecolor='black', alpha=0.8)
            
            # Mark significance
            for bar, sig in zip(bars, p_data['Significant']):
                if sig:
                    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
                            '*', va='center', fontweight='bold')

            ax.set_title(player)
            ax.set_xlim(-0.1, 1.0)
            
            if idx == 0:
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='#ff9999', label='Attacking'),
                                 Patch(facecolor='#66b3ff', label='Creative')]
                ax.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        outfile = 'H1_metrics_correlation.png'
        plt.savefig(outfile, dpi=300)
        print(f"Plot saved: {outfile}")
        plt.close()
    
    def phase2_rating_to_outcome(self):
        print("\n--- Phase 2: Aggregate Metrics vs Match Result (ANOVA) ---")
        
        match_agg = self.create_match_aggregates()
        metrics = ['Total_npxG', 'Total_SCA', 'Total_Goals', 'Avg_Rating', 'Total_PrgCarries']
        
        anova_results = []
        for metric in metrics:
            if metric in match_agg.columns:
                groups = [match_agg[match_agg['result'].str.contains(r, na=False)][metric].dropna() for r in ['W', 'D', 'L']]
                
                if all(len(g) > 0 for g in groups):
                    f, p = f_oneway(*groups)
                    sig = "*" if p < 0.05 else ""
                    print(f"{metric}: F={f:.3f}, p={p:.4f} {sig}")
                    anova_results.append({'Metric': metric, 'Significant': p < 0.05})
        
        self.plot_phase2_results(match_agg, anova_results)

    def create_match_aggregates(self):
        agg_rules = {
            'npxg': 'sum', 
            'sca': 'sum', 
            'Gls': 'sum', 
            'Rating': 'mean', 
            'progressive_carries': 'sum', 
            'result': 'first'
        }
        # Filter rules based on available columns
        agg_rules = {k: v for k, v in agg_rules.items() if k in self.df.columns}
        
        match_agg = self.df.groupby(['Date', 'Opponent']).agg(agg_rules).reset_index()
        
        rename_map = {
            'npxg': 'Total_npxG', 
            'sca': 'Total_SCA', 
            'Gls': 'Total_Goals', 
            'Rating': 'Avg_Rating', 
            'progressive_carries': 'Total_PrgCarries'
        }
        return match_agg.rename(columns=rename_map)

    def plot_phase2_results(self, match_agg, anova_results):
        metrics_to_plot = ['Total_npxG', 'Total_SCA', 'Total_PrgCarries', 'Avg_Rating']
        available_metrics = [m for m in metrics_to_plot if m in match_agg.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(20, 5)) 
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx] if len(available_metrics) > 1 else axes
            
            data_to_plot = []
            labels = ['W', 'D', 'L']
            colors = ['#90EE90', '#FFFFE0', '#FFB6C1']
            
            for r in labels:
                subset = match_agg[match_agg['result'].str.contains(r, na=False)][metric].dropna()
                data_to_plot.append(subset.values)
            
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(metric)
            ax.grid(alpha=0.3)
            
            # Annotate significance
            is_sig = next((r['Significant'] for r in anova_results if r['Metric'] == metric), False)
            if is_sig:
                ax.text(0.5, 0.9, 'p < 0.05', transform=ax.transAxes, ha='center', 
                        color='red', fontweight='bold')

        plt.tight_layout()
        outfile = 'H2_team_impact.png'
        plt.savefig(outfile, dpi=300)
        print(f"Plot saved: {outfile}")
        plt.close()

def main():
    data_file = 'fbref_sofascore_merged_by_date_only.csv'
    try:
        tester = BarcelonaHypothesisTesting(data_file)
        tester.check_assumptions()
        tester.phase1_metrics_to_rating()
        tester.phase2_rating_to_outcome()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()