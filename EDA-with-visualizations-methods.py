import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import warnings
import os

warnings.filterwarnings('ignore')

# Visualization settings
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class BarcelonaEDA:
    def __init__(self, data_file):
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"File not found: {data_file}")
            
        self.df = pd.read_csv(data_file)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Metric definitions
        self.attacking_metrics = ['Gls', 'Ast', 'shots', 'shots_on_target', 'xg', 'npxg', 'xg_assist']
        self.creative_metrics = ['sca', 'gca', 'progressive_passes', 'progressive_carries', 
                                'passes_completed', 'passes_pct', 'take_ons', 'take_ons_won']
        self.defensive_metrics = ['tackles', 'interceptions', 'blocks']
        self.discipline_metrics = ['cards_yellow', 'cards_red']
        
        print(f"Loaded {len(self.df)} rows. Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
    
    def data_overview(self):
        print("\n--- Data Overview ---")
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("\nMissing Values:")
            print(missing[missing > 0])
        
        print("\nDescriptive Statistics:")
        print(self.df[['Rating', 'Min', 'Gls', 'Ast', 'xg', 'sca']].describe())
    
    def player_summary(self):
        print("\n--- Player Summaries ---")
        for player in self.df['Player'].unique():
            player_data = self.df[self.df['Player'] == player]
            starts = (player_data['Started'] == 'Y').sum()
            
            print(f"[{player}]")
            print(f"  Matches: {len(player_data)} (Starts: {starts})")
            print(f"  Rating: {player_data['Rating'].mean():.2f} +/- {player_data['Rating'].std():.2f}")
            print(f"  G/A: {player_data['Gls'].sum()} / {player_data['Ast'].sum()}")
            print(f"  Avg Min: {player_data['Min'].mean():.1f}")
    
    def match_result_analysis(self):
        print("\n--- Performance by Result ---")
        print(self.df['result'].value_counts())
        
        print("\nAverage Rating by Result:")
        for result_type in ['W', 'D', 'L']:
            mask = self.df['result'].str.contains(result_type, na=False)
            if mask.sum() > 0:
                avg = self.df.loc[mask, 'Rating'].mean()
                std = self.df.loc[mask, 'Rating'].std()
                print(f"  {result_type}: {avg:.2f} (std: {std:.2f}) n={mask.sum()}")
    
    def correlation_analysis(self):
        print("\n--- Correlation Analysis ---")
        all_metrics = (self.attacking_metrics + self.creative_metrics + self.discipline_metrics)
        
        pearson_correlations = []
        
        print("Metric correlations with Rating (Pearson | Spearman):")
        for metric in all_metrics:
            if metric in self.df.columns:
                mask = self.df[metric].notna() & self.df['Rating'].notna()
                if mask.sum() > 10:
                    corr_p, _ = stats.pearsonr(self.df.loc[mask, metric], self.df.loc[mask, 'Rating'])
                    corr_s, _ = spearmanr(self.df.loc[mask, metric], self.df.loc[mask, 'Rating'])
                    
                    pearson_correlations.append({
                        'Metric': metric, 'Pearson': corr_p, 'Spearman': corr_s
                    })
                    print(f"  {metric:25s}: {corr_p:6.3f} | {corr_s:6.3f}")
        
        print("\nImpact Profile (Attacking vs Creative):")
        for player in self.df['Player'].unique():
            player_data = self.df[self.df['Player'] == player]
            
            # Helper to get avg correlation for a list of metrics
            def get_avg_corr(metrics, data):
                corrs = []
                for m in metrics:
                    if m in data.columns:
                        mask = data[m].notna() & data['Rating'].notna()
                        if mask.sum() > 5:
                            c, _ = stats.pearsonr(data.loc[mask, m], data.loc[mask, 'Rating'])
                            corrs.append(abs(c))
                return np.mean(corrs) if corrs else 0

            avg_att = get_avg_corr(self.attacking_metrics, player_data)
            avg_crt = get_avg_corr(self.creative_metrics, player_data)
            
            profile = "Balanced"
            if avg_att > avg_crt * 1.1: profile = "Attacking Dominant"
            elif avg_crt > avg_att * 1.1: profile = "Creative Dominant"
            
            print(f"  {player}: Att={avg_att:.3f}, Cre={avg_crt:.3f} -> {profile}")

        return pd.DataFrame(pearson_correlations).sort_values('Pearson', ascending=False)
    
    def outlier_detection(self):
        print("\n--- Outlier Detection (IQR) ---")
        metrics_to_check = ['Rating', 'Gls', 'Ast', 'xg', 'sca']
        
        for metric in metrics_to_check:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                bounds = (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
                
                outliers = self.df[(self.df[metric] < bounds[0]) | (self.df[metric] > bounds[1])]
                
                print(f"[{metric}] Bounds: {bounds[0]:.2f} - {bounds[1]:.2f}. Outliers found: {len(outliers)}")
                if not outliers.empty:
                    for _, row in outliers[['Player', 'Date', 'Opponent', metric]].head(3).iterrows():
                        print(f"    {row['Player']} vs {row['Opponent']}: {row[metric]}")
    
    def visualize_ratings(self):
        print("\nGenerating visualizations...")
        self.plot_rating_distributions_kde()
        self.plot_correlation_heatmap()
        self.plot_violin_by_result()
        self.plot_scatter_attacking_per_player()
        self.plot_scatter_creative_per_player()
        self.plot_discipline_analysis()
        self.plot_player_comparison()
        self.plot_attacking_vs_creative_impact()
        self.plot_rating_trends_over_season()
    
    def plot_rating_distributions_kde(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Player Rating Distributions (KDE)', fontsize=16)
        
        players = self.df['Player'].unique()
        for idx, player in enumerate(players):
            if idx >= 4: break
            ax = axes[idx // 2, idx % 2]
            player_data = self.df[self.df['Player'] == player]
            
            ax.hist(player_data['Rating'], bins=15, edgecolor='black', alpha=0.5, 
                   color='skyblue', density=True)
            
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(player_data['Rating'].dropna())
                x_range = np.linspace(player_data['Rating'].min(), player_data['Rating'].max(), 100)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2)
            except:
                pass # Skip KDE if insufficient data
            
            ax.axvline(player_data['Rating'].mean(), color='darkred', linestyle='--')
            ax.set_title(f'{player} (n={len(player_data)})')
            ax.set_xlabel('Rating')
        
        plt.tight_layout()
        plt.savefig('01_rating_distributions_kde.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 01_rating_distributions_kde.png")
    
    def plot_correlation_heatmap(self):
        key_metrics = ['Rating', 'Gls', 'Ast', 'xg', 'xg_assist', 'sca', 'gca',
                      'progressive_passes', 'progressive_carries', 'passes_pct', 'touches']
        available = [m for m in key_metrics if m in self.df.columns]
        
        corr_matrix = self.df[available].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, vmin=-1, vmax=1, ax=ax)
        
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        plt.savefig('02_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 02_correlation_heatmap.png")
    
    def plot_violin_by_result(self):
        metrics = ['Rating', 'Gls', 'Ast', 'xg', 'sca', 'progressive_passes']
        available = [m for m in metrics if m in self.df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available):
            ax = axes[idx]
            plot_data = []
            
            for result_type in ['W', 'D', 'L']:
                mask = self.df['result'].str.contains(result_type, na=False) & self.df[metric].notna()
                if mask.sum() > 0:
                    temp_df = pd.DataFrame({'Value': self.df.loc[mask, metric], 'Result': result_type})
                    plot_data.append(temp_df)
            
            if plot_data:
                combined = pd.concat(plot_data, ignore_index=True)
                parts = ax.violinplot([combined[combined['Result'] == r]['Value'].values 
                                      for r in ['W', 'D', 'L'] if r in combined['Result'].unique()],
                                     showmeans=True, showmedians=True)
                
                colors = ['lightgreen', 'lightyellow', 'lightcoral']
                for pc, color in zip(parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                
                ax.set_xticks(range(1, len(combined['Result'].unique()) + 1))
                ax.set_xticklabels([r for r in ['W', 'D', 'L'] if r in combined['Result'].unique()])
                ax.set_title(metric)
        
        plt.tight_layout()
        plt.savefig('03_violin_plots_by_result.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 03_violin_plots_by_result.png")
    
    def plot_scatter_attacking_per_player(self):
        self._plot_scatter_grid(self.attacking_metrics, 'attacking', 'red')

    def plot_scatter_creative_per_player(self):
        self._plot_scatter_grid(self.creative_metrics, 'creative', 'blue')
        
    def _plot_scatter_grid(self, metrics, category_name, color):
        """Helper to reduce code duplication for scatter plots"""
        available = [m for m in metrics if m in self.df.columns]
        players = self.df['Player'].unique()
        
        for player in players:
            player_data = self.df[self.df['Player'] == player]
            fig, axes = plt.subplots(2, 4, figsize=(24, 12))
            fig.suptitle(f'{player} - {category_name.title()} Metrics vs Rating', fontsize=16)
            axes = axes.flatten()
            
            for idx, metric in enumerate(available[:8]):
                ax = axes[idx]
                mask = player_data[metric].notna() & player_data['Rating'].notna()
                
                if mask.sum() > 5:
                    x = player_data.loc[mask, metric]
                    y = player_data.loc[mask, 'Rating']
                    
                    ax.scatter(x, y, alpha=0.6, s=60, color=color, edgecolors='black')
                    
                    # Trend line
                    if len(x) > 2:
                        z = np.polyfit(x, y, 1)
                        p = np.poly1d(z)
                        ax.plot(x, p(x), color=f"dark{color}", linestyle='--')
                    
                    corr_p, _ = stats.pearsonr(x, y)
                    ax.set_xlabel(metric)
                    ax.set_ylabel('Rating')
                    ax.set_title(f'{metric} (r={corr_p:.2f})')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', transform=ax.transAxes)
            
            # Clean up empty axes
            for idx in range(len(available), 8):
                fig.delaxes(axes[idx])
                
            plt.tight_layout()
            filename = f'04_{player}_{category_name}_scatter.png' if category_name == 'attacking' else f'05_{player}_{category_name}_scatter.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {filename}")

    def plot_discipline_analysis(self):
        if 'cards_yellow' not in self.df.columns: return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Discipline Analysis', fontsize=16)
        
        # 1. Rating Impact
        ax = axes[0, 0]
        mask = self.df['cards_yellow'].notna() & self.df['Rating'].notna()
        with_card = self.df.loc[mask & (self.df['cards_yellow'] > 0), 'Rating']
        without_card = self.df.loc[mask & (self.df['cards_yellow'] == 0), 'Rating']
        
        if len(with_card) > 0 and len(without_card) > 0:
            ax.boxplot([without_card, with_card], labels=['No Card', 'Yellow Card'], patch_artist=True)
            t_stat, p_val = stats.ttest_ind(with_card, without_card)
            ax.set_title(f'Rating Impact (p={p_val:.3f})')
        
        # 2. Total Cards
        ax = axes[0, 1]
        players = self.df['Player'].unique()
        yellow_counts = [self.df[self.df['Player'] == p]['cards_yellow'].sum() for p in players]
        ax.bar(players, yellow_counts, color='yellow', edgecolor='black', alpha=0.7)
        ax.set_title('Total Yellow Cards')
        
        # 3. Rating when Booked
        ax = axes[1, 0]
        ratings_when_booked = []
        labels = []
        for player in players:
            pd_sub = self.df[self.df['Player'] == player]
            mask = (pd_sub['cards_yellow'] > 0) & pd_sub['Rating'].notna()
            if mask.sum() > 0:
                ratings_when_booked.append(pd_sub.loc[mask, 'Rating'].mean())
                labels.append(player)
        
        if ratings_when_booked:
            ax.bar(labels, ratings_when_booked, color='orange', edgecolor='black')
            ax.set_title('Avg Rating when Booked')
            ax.axhline(self.df['Rating'].mean(), color='red', linestyle='--')
            
        # 4. Scatter
        ax = axes[1, 1]
        mask = self.df['cards_yellow'].notna() & self.df['Rating'].notna()
        ax.scatter(self.df.loc[mask, 'cards_yellow'], self.df.loc[mask, 'Rating'], alpha=0.5)
        ax.set_title('Cards vs Rating Scatter')

        plt.tight_layout()
        plt.savefig('06_discipline_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 06_discipline_analysis.png")

    def plot_player_comparison(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        players = self.df['Player'].unique()
        
        metrics = [
            ('Rating', 'mean', 'Average Rating', 'skyblue'),
            ('Gls', 'sum', 'Total Goals', 'green'),
            ('Ast', 'sum', 'Total Assists', 'orange'),
            ('sca', 'mean', 'Avg SCA', 'purple')
        ]
        
        for idx, (col, agg, title, color) in enumerate(metrics):
            if col not in self.df.columns: continue
            ax = axes[idx // 2, idx % 2]
            
            if agg == 'mean':
                values = [self.df[self.df['Player'] == p][col].mean() for p in players]
            else:
                values = [self.df[self.df['Player'] == p][col].sum() for p in players]
                
            ax.bar(players, values, color=color, alpha=0.8, edgecolor='black')
            ax.set_title(title)
            
        plt.tight_layout()
        plt.savefig('07_player_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 07_player_comparison.png")


    def plot_attacking_vs_creative_impact(self):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        players = self.df['Player'].unique()
        
        for idx, player in enumerate(players):
            if idx >= 4: break
            ax = axes[idx // 2, idx % 2]
            player_data = self.df[self.df['Player'] == player]
            
            def get_avg_corr(metrics):
                corrs = []
                for m in metrics:
                    if m in player_data.columns:
                        mask = player_data[m].notna() & player_data['Rating'].notna()
                        if mask.sum() > 5:
                            c, _ = stats.pearsonr(player_data.loc[mask, m], player_data.loc[mask, 'Rating'])
                            corrs.append(abs(c))
                return np.mean(corrs) if corrs else 0
            
            avg_att = get_avg_corr(self.attacking_metrics)
            avg_cre = get_avg_corr(self.creative_metrics)
            
            ax.bar(['Attacking', 'Creative'], [avg_att, avg_cre], 
                  color=['red', 'blue'], alpha=0.7, edgecolor='black')
            ax.set_title(f'{player} Impact Profile')
            ax.set_ylabel('Mean Absolute Correlation')
            
        plt.tight_layout()
        plt.savefig('08_attacking_vs_creative_impact.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 08_attacking_vs_creative_impact.png")
        
    def plot_rating_trends_over_season(self):
        """Plot rating trends over time for each player"""
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Rating Trends Over Season', fontsize=16)
        axes = axes.flatten()
        
        players = self.df['Player'].unique()
        
        for idx, player in enumerate(players):
            if idx >= 4: break
            ax = axes[idx]
            player_data = self.df[self.df['Player'] == player].sort_values('Date')
            
            # Plot ratings
            ax.plot(player_data['Date'], player_data['Rating'], 
                marker='o', linestyle='-', alpha=0.6, markersize=4)
            
            # Rolling average (7-game window)
            if len(player_data) > 7:
                rolling_avg = player_data['Rating'].rolling(window=7, min_periods=3).mean()
                ax.plot(player_data['Date'], rolling_avg, 
                    color='red', linewidth=2, label='7-game avg')
            
            # Overall average line
            avg_rating = player_data['Rating'].mean()
            ax.axhline(avg_rating, color='green', linestyle='--', 
                    alpha=0.7, label=f'Season avg: {avg_rating:.2f}')
            
            ax.set_title(f'{player}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Rating')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig('09_rating_trends_over_season.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved: 09_rating_trends_over_season.png")

def main():
    data_file = 'fbref_sofascore_merged_by_date_only.csv'
    try:
        eda = BarcelonaEDA(data_file)
        eda.data_overview()
        eda.player_summary()
        eda.match_result_analysis()
        eda.correlation_analysis()
        eda.outlier_detection()
        eda.visualize_ratings()
        print("\nAnalysis complete.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()