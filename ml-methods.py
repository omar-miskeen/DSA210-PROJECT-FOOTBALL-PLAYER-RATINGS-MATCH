import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('results', exist_ok=True)


def load_and_prep_data(filepath):
    """Load data and create basic features"""
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create simple features
    if 'passes_completed' in df.columns and 'touches' in df.columns:
        df['pass_efficiency'] = df['passes_completed'] / (df['touches'] + 1)
    
    if 'progressive_passes' in df.columns and 'progressive_carries' in df.columns:
        df['progressive_actions'] = df['progressive_passes'] + df['progressive_carries']
    
    if 'tackles' in df.columns and 'interceptions' in df.columns:
        df['defensive_work'] = df['tackles'] + df['interceptions']
    
    if 'result' in df.columns:
        df['result_type'] = df['result'].str.extract(r'([WDL])', expand=False)
    
    return df


def get_features_regression(df):
    """Get features for regression - keeps Gls/Ast"""
    exclude = ['Date', 'Player', 'Team', 'Opponent', 'result', 'result_type',
               'Rating', 'Started', 'Position', 'venue', 'Competition', 'Venue']
    
    features = [col for col in df.columns 
                if col not in exclude and df[col].dtype in ['int64', 'float64']]
    return features


def get_features_classification(df):
    """Get features for classification - excludes Gls/Ast"""
    exclude = ['Date', 'Player', 'Team', 'Opponent', 'result', 'result_type',
               'Rating', 'Started', 'Position', 'venue', 'Competition', 'Venue',
               'Gls', 'Ast', 'Goal_Contribution']
    
    features = [col for col in df.columns 
                if col not in exclude and df[col].dtype in ['int64', 'float64']]
    return features


def run_regression(df, split_type='random'):
    """Run regression to predict Rating"""
    print(f"\n{'='*60}")
    print(f"REGRESSION - Rating Prediction ({split_type.upper()} SPLIT)")
    print(f"{'='*60}")
    
    # Get data
    features = get_features_regression(df)
    X = df[features].fillna(0)
    y = df['Rating'].fillna(df['Rating'].median())
    
    # Split data
    if split_type == 'temporal':
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Test models with full hyperparameter grids
    results = {}
    models = {}
    
    # Ridge - full grid
    ridge = GridSearchCV(Ridge(), {'alpha': [0.1, 1.0, 10.0, 100.0]}, cv=5)
    ridge.fit(X_train_scaled, y_train)
    y_pred = ridge.predict(X_test_scaled)
    results['Ridge'] = r2_score(y_test, y_pred)
    models['Ridge'] = ridge.best_estimator_
    
    # Lasso - full grid
    lasso = GridSearchCV(Lasso(max_iter=10000), {'alpha': [0.01, 0.1, 1.0, 10.0]}, cv=5)
    lasso.fit(X_train_scaled, y_train)
    y_pred = lasso.predict(X_test_scaled)
    results['Lasso'] = r2_score(y_test, y_pred)
    models['Lasso'] = lasso.best_estimator_
    
    # Random Forest - full grid
    rf = GridSearchCV(RandomForestRegressor(random_state=42), 
                      {'n_estimators': [50, 100, 150], 
                       'max_depth': [5, 10, 15],
                       'min_samples_leaf': [3, 5, 8]}, cv=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results['Random Forest'] = r2_score(y_test, y_pred)
    models['Random Forest'] = rf.best_estimator_
    
    # XGBoost - full grid
    xgb_model = GridSearchCV(xgb.XGBRegressor(random_state=42),
                             {'n_estimators': [50, 100, 150],
                              'max_depth': [3, 5, 7],
                              'learning_rate': [0.01, 0.1, 0.3]}, cv=5)
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)
    results['XGBoost'] = r2_score(y_test, y_pred)
    models['XGBoost'] = xgb_model.best_estimator_
    
    # Print results
    print("\nResults:")
    for model, r2 in results.items():
        print(f"  {model:20s} R² = {r2:.4f}")
    
    best = max(results, key=results.get)
    print(f"\nBest: {best} (R² = {results[best]:.4f})")
    
    # Plot model comparison
    plt.figure(figsize=(10, 6))
    plt.barh(list(results.keys()), list(results.values()), color='teal', edgecolor='black')
    plt.xlabel('R² Score')
    plt.title(f'Model Comparison - {split_type.upper()} Split')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/regression_comparison_{split_type}.png', dpi=150)
    plt.close()
    
    # Plot feature importance for best model
    best_model = models[best]
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_)  # Use absolute value for magnitude        
    if importances is not None:
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(15), importances[indices], color='green', edgecolor='black')
        plt.yticks(range(15), [features[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Features - {best} ({split_type.upper()})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'results/regression_features_{split_type}.png', dpi=150)
        plt.close()
    
    return results


def run_classification(df, split_type='random'):
    """Run classification to predict Win/Draw/Loss"""
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION - Outcome Prediction ({split_type.upper()} SPLIT)")
    print(f"{'='*60}")
    
    # Get data
    df_clean = df.dropna(subset=['result_type'])
    features = get_features_classification(df_clean)
    X = df_clean[features].fillna(0)
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df_clean['result_type'])
    
    # Split
    if split_type == 'temporal':
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"Baseline (predict W): {np.sum(y_test == 2) / len(y_test):.2%}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote_variants = {
        'SMOTE': SMOTE(random_state=42, sampling_strategy='not majority'),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42, sampling_strategy='not majority'),
        'SVMSMOTE': SVMSMOTE(random_state=42, sampling_strategy='not majority'),
    }
    
    best_smote = None
    best_acc = 0
    best_X_train = None
    best_y_train = None
    
    print("\nTesting SMOTE variants:")
    for smote_name, smote in smote_variants.items():
        try:
            X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
            
            # Quick test
            test_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            test_rf.fit(X_resampled, y_resampled)
            y_pred = test_rf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            print(f"  {smote_name:20s} Accuracy = {acc:.2%}")
            
            if acc > best_acc:
                best_acc = acc
                best_smote = smote_name
                best_X_train = X_resampled
                best_y_train = y_resampled
        except Exception as e:
            print(f"  {smote_name:20s} Failed: {str(e)[:50]}")
    
    print(f"\nBest SMOTE: {best_smote} ({best_acc:.2%})")
    
    # Test models with full hyperparameter grids
    results = {}
    models = {}
    
    # Random Forest - full grid
    rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced'),
                      {'n_estimators': [100, 150, 200],
                       'max_depth': [10, 14, 18],
                       'min_samples_leaf': [5, 8, 10]}, cv=5)
    rf.fit(best_X_train, best_y_train)
    y_pred = rf.predict(X_test_scaled)
    results['Random Forest'] = accuracy_score(y_test, y_pred)
    models['Random Forest'] = (rf.best_estimator_, y_pred)
    
    # XGBoost - full grid
    xgb_model = GridSearchCV(xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                             {'n_estimators': [100, 150],
                              'max_depth': [5, 7, 10],
                              'learning_rate': [0.01, 0.1]}, cv=5)
    xgb_model.fit(best_X_train, best_y_train)
    y_pred = xgb_model.predict(X_test_scaled)
    results['XGBoost'] = accuracy_score(y_test, y_pred)
    models['XGBoost'] = (xgb_model.best_estimator_, y_pred)
    
    # Print results
    print("\nResults:")
    for model, acc in results.items():
        print(f"  {model:20s} Accuracy = {acc:.2%}")
    
    best = max(results, key=results.get)
    print(f"\nBest: {best} ({results[best]:.2%})")
    
    # Classification report
    best_model, y_pred = models[best]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap='Blues')
    plt.title(f'Confusion Matrix - {best} ({split_type.upper()})')
    plt.tight_layout()
    plt.savefig(f'results/classification_confusion_{split_type}.png', dpi=150)
    plt.close()
    
    # Plot feature importance
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(15), importances[indices], color='purple', edgecolor='black')
        plt.yticks(range(15), [features[i] for i in indices])
        plt.xlabel('Importance')
        plt.title(f'Top 15 Features - {best} ({split_type.upper()})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'results/classification_features_{split_type}.png', dpi=150)
        plt.close()
    
    # Plot model comparison
    plt.figure(figsize=(10, 6))
    plt.barh(list(results.keys()), list(results.values()), color='orange', edgecolor='black')
    plt.xlabel('Accuracy')
    plt.title(f'Model Comparison - {split_type.upper()} Split')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'results/classification_comparison_{split_type}.png', dpi=150)
    plt.close()
    
    return results


def main():
    print("\n" + "="*60)
    print("SIMPLE BARCELONA ML ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_and_prep_data('fbref_sofascore_merged_by_date_only.csv')
    print(f"\nLoaded {len(df)} matches")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Run both split types
    for split_type in ['temporal', 'random']:
        print(f"\n\n{'#'*60}")
        print(f"TESTING {split_type.upper()} SPLIT")
        print(f"{'#'*60}")
        
        # Regression
        reg_results = run_regression(df, split_type)
        
        # Classification
        clf_results = run_classification(df, split_type)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":

    main()
