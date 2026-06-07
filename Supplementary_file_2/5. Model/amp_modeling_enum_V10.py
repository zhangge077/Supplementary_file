#!/usr/bin/env python3
"""
AMP (Antimicrobial Peptide) Enumerative Modeling Program - V10
Enhanced Version with LOOCV, Feature Selection, LDA SHAP Analysis
+ SCI Publication-Ready Analysis for Best Model

V10 NEW FEATURES (SCI Publication Analysis):
1. Comprehensive Feature Importance Ranking (multiple methods)
2. Permutation Importance Analysis with statistical tests
3. Partial Dependence Plots (PDP)
4. Calibration Curves (reliability diagrams)
5. Decision Boundary Visualization
6. Learning Curves
7. Cross-Validation Stability Analysis
8. Statistical Significance Tests (McNemar's test)
9. Publication-Ready Figures with proper formatting
10. Detailed Results Tables for SCI Papers

Features from V9:
1. LOOCV (Leave-One-Out Cross-Validation) for small datasets
2. Multiple feature selection methods (SelectKBest, Mutual Info, RFE, etc.)
3. LDA SHAP explanation using KernelExplainer
4. Feature counts: 5 and 10 only
5. Comprehensive model evaluation with multiple metrics
6. SHAP analysis for interpretable models including LDA
7. Two-class classification based on Exp_Log2_MIC:
   - Low (<=3)
   - High (>3)
"""

import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif, RFE, RFECV,
    SelectFromModel, SelectPercentile, chi2
)
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, cross_val_predict,
    train_test_split, LeaveOneOut, KFold, learning_curve,
    RepeatedStratifiedKFold
)
from sklearn.inspection import permutation_importance
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier,
    VotingClassifier, StackingClassifier, HistGradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, auc, average_precision_score,
    make_scorer, get_scorer, balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, log_loss
)
from sklearn.pipeline import Pipeline
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
import warnings
warnings.filterwarnings('ignore')

import pickle
import json
import os
from datetime import datetime
from itertools import product

# Try to import XGBoost, LightGBM, and CatBoost
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not available.")

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available.")

# Try to import SHAP
SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP imported successfully.")
except ImportError:
    print("Warning: SHAP not available. SHAP analysis will be skipped.")

# Create output directories (V10_results)
OUTPUT_DIR = 'V10_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'best_model_plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'shap_plots'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'loocv_results'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'sci_plots'), exist_ok=True)  # NEW: SCI publication plots
os.makedirs(os.path.join(OUTPUT_DIR, 'sci_tables'), exist_ok=True)  # NEW: SCI tables


def safe_savefig(fig, *paths, **kwargs):
    """Safely save matplotlib figures to disk with cross-platform path handling."""
    import matplotlib
    matplotlib.use('Agg')  # Ensure Agg backend

    # Convert all path parts to use forward slashes for cross-platform compatibility
    safe_paths = [str(p).replace('\\', '/') for p in paths]

    # Save with dpi parameter if provided
    dpi = kwargs.get('dpi', 300)

    # Try saving with different path formats
    for path in safe_paths:
        try:
            # Ensure directory exists
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            return path
        except Exception as e:
            # If first attempt fails, try alternative
            print(f"Warning: Failed to save to {path}: {e}")
            continue

    # If all attempts fail, use a simple filename
    fallback_path = safe_paths[-1] if safe_paths else 'output.png'
    try:
        fig.savefig(fallback_path, dpi=dpi, bbox_inches='tight')
    except:
        # Last resort: save to current directory
        fig.savefig('output.png', dpi=dpi, bbox_inches='tight')

    return fallback_path

# Set style for all plots - Publication quality
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False
})


def read_excel_robust(data_path):
    """Robustly read Excel file with multiple fallback methods."""
    import os

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    file_size = os.path.getsize(data_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {data_path}")

    print(f"  File size: {file_size} bytes")

    try:
        print("  Trying openpyxl engine...")
        df = pd.read_excel(data_path, engine='openpyxl')
        if df is not None and not df.empty:
            print(f"  Success! Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"  openpyxl failed: {type(e).__name__}: {e}")

    try:
        print("  Trying xlrd engine...")
        df = pd.read_excel(data_path, engine='xlrd')
        if df is not None and not df.empty:
            print(f"  Success! Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"  xlrd failed: {type(e).__name__}: {e}")

    try:
        print("  Trying to read as CSV...")
        df = pd.read_csv(data_path)
        if df is not None and not df.empty:
            print(f"  Success as CSV! Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"  CSV attempt failed: {type(e).__name__}: {e}")

    for sep in [';', '\t', '|']:
        try:
            print(f"  Trying CSV with separator '{sep}'...")
            df = pd.read_csv(data_path, sep=sep)
            if df is not None and not df.empty:
                print(f"  Success! Loaded {len(df)} rows, {len(df.columns)} columns")
                return df
        except Exception as e:
            continue

    try:
        print("  Trying openpyxl with data_only=True...")
        df = pd.read_excel(data_path, engine='openpyxl', data_only=True)
        if df is not None and not df.empty:
            print(f"  Success! Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
    except Exception as e:
        print(f"  data_only attempt failed: {type(e).__name__}: {e}")

    raise ValueError(f"Cannot read file with any known method: {data_path}")


def categorize_mic(value):
    """Convert Exp_Log2_MIC value to two-class category based on MIC thresholds."""
    if value <= 3.0:
        return 'Low'
    else:
        return 'High'


class AMPModelingV10:
    """AMP Modeling Class with LOOCV, Feature Selection, LDA SHAP, and SCI Analysis - V10"""

    def __init__(self, data_path):
        """Initialize modeler"""
        self.df = read_excel_robust(data_path)
        self.X_raw = None
        self.y = None
        self.feature_cols = None
        self.scaler = StandardScaler()
        self.all_results = []
        self.loocv_results = []
        self.reduction_results = {}
        self.label_encoder = LabelEncoder()
        self.best_model_name = None
        self.feature_selection_methods = {}
        self.sci_results = {}  # NEW: SCI analysis results

    def preprocess(self):
        """Data preprocessing"""
        print("=" * 80)
        print("STEP 1: DATA PREPROCESSING (V10 - Two-Class + SCI Analysis)")
        print("=" * 80)

        self.df['MIC_Category'] = self.df['Exp_Log2_MIC'].apply(categorize_mic)

        print(f"\n[Dataset Information]")
        print(f"  Samples: {len(self.df)}")
        print(f"  Original Features: {len(self.df.columns)}")

        mic_dist = self.df['MIC_Category'].value_counts()
        print(f"\n[MIC Category Distribution - Two Classes]")
        for cat in ['Low', 'High']:
            count = mic_dist.get(cat, 0)
            print(f"  {cat:8s}: {count:3d} samples ({count/len(self.df)*100:5.1f}%)")

        exclude_cols = ['Name', 'Sequences', 'Exp_Log2_MIC', 'MIC_Category']
        self.feature_cols = [
            col for col in self.df.columns
            if col not in exclude_cols and self.df[col].dtype in ['int64', 'float64']
        ]
        print(f"\n[Features Used]: {len(self.feature_cols)}")

        X = self.df[self.feature_cols].copy()
        missing = X.isnull().sum().sum()
        if missing > 0:
            print(f"\n[Missing Values]: {missing} filled with median")
            X = X.fillna(X.median())

        self.X_raw = X.values
        self.X_raw_df = X
        self.y = self.df['MIC_Category']
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        self.class_names = list(self.label_encoder.classes_)
        self.n_classes = len(self.class_names)

        self._save_dataset_summary()
        self._generate_data_visualizations()

        return self

    def _save_dataset_summary(self):
        """Save dataset summary"""
        summary = {
            'n_samples': len(self.df),
            'n_features': len(self.feature_cols),
            'feature_names': self.feature_cols,
            'class_distribution': self.df['MIC_Category'].value_counts().to_dict(),
            'missing_values': int(self.X_raw_df.isnull().sum().sum()),
            'n_classes': self.n_classes,
            'class_names': self.class_names,
            'classification_type': 'two-class',
            'thresholds': {
                'Low': '<=3',
                'High': '>3'
            },
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(OUTPUT_DIR, 'dataset_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        dist_df = pd.DataFrame({
            'Category': self.class_names,
            'Threshold': ['<=3', '>3'][:self.n_classes],
            'Count': [self.df['MIC_Category'].value_counts().get(c, 0) for c in self.class_names],
            'Percentage': [f"{self.df['MIC_Category'].value_counts().get(c, 0)/len(self.df)*100:.1f}%" for c in self.class_names]
        })
        dist_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'class_distribution.csv'), index=False)
        dist_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'class_distribution.xlsx'), index=False)

    def _generate_data_visualizations(self):
        """Generate initial data visualizations"""
        print("\n[Generating Data Visualizations...]")

        # Class Distribution
        fig, ax = plt.subplots(figsize=(10, 10))
        colors = ['#2ecc71', '#e74c3c'][:self.n_classes]
        dist = [self.df['MIC_Category'].value_counts().get(c, 0) for c in self.class_names]
        labels_with_count = [f'{c}\n({count})' for c, count in zip(self.class_names, dist)]
        wedges, texts, autotexts = ax.pie(dist, labels=labels_with_count, autopct='%1.1f%%',
                                          colors=colors, explode=[0.02]*self.n_classes,
                                          shadow=True, startangle=90)
        ax.set_title('MIC Category Distribution (Two-Class)\nLow (<=3) | High (>3)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_class_distribution.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_class_distribution.pdf'))
        plt.close()

        # Feature Distribution
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        for i, col in enumerate(self.feature_cols[:16]):
            ax = axes[i]
            data = self.X_raw_df[col].dropna()
            ax.hist(data, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
            ax.set_title(col[:20], fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_feature_distributions.png'), dpi=300)
        plt.close()

        # Box plots
        fig, ax = plt.subplots(figsize=(16, 8))
        top_features = self.feature_cols[:20]
        data_to_plot = [self.X_raw_df[f].dropna().values for f in top_features]
        bp = ax.boxplot(data_to_plot, labels=[f[:15] for f in top_features], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_title('Feature Value Distributions (Box Plot)', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_feature_boxplots.png'), dpi=300)
        plt.close()

        # Correlation Heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        corr_matrix = self.X_raw_df.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Heatmap', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_correlation_heatmap.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_correlation_heatmap.pdf'))
        plt.close()

        # PCA 2D projection
        X_scaled = self.scaler.fit_transform(self.X_raw)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['#2ecc71', '#e74c3c'][:self.n_classes]
        markers = ['o', 's'][:self.n_classes]
        for i, cls in enumerate(self.class_names):
            mask = self.y == cls
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=colors[i], label=cls, alpha=0.7, s=100, marker=markers[i])
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title('PCA 2D Projection of Dataset (Two-Class)', fontsize=14)
        ax.legend(title='MIC Category')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', '00_pca_2d_projection.png'), dpi=300)
        plt.close()

        print(f"  Generated 5 initial visualizations")

    def scale_features(self):
        """Feature scaling"""
        print("\n" + "=" * 80)
        print("STEP 2: FEATURE SCALING")
        print("=" * 80)

        self.X_scaled = self.scaler.fit_transform(self.X_raw)
        print(f"Scaled features: {self.X_scaled.shape}")

        return self

    def feature_selection(self):
        """Multiple feature selection methods"""
        print("\n" + "=" * 80)
        print("STEP 3: FEATURE SELECTION METHODS")
        print("=" * 80)

        # SelectKBest with f_classif
        print("\n[1] SelectKBest (f_classif)...")
        for k in [5, 10]:
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(self.X_scaled, self.y_encoded)
            selected_features = [self.feature_cols[i] for i in selector.get_support(indices=True)]
            self.reduction_results[f'selectkbest_f_classif_{k}'] = {
                'X': X_selected,
                'n_components': k,
                'description': f'SelectKBest-f_classif-{k}',
                'model': selector,
                'features': selected_features,
                'scores': selector.scores_.tolist(),
                'method': 'selectkbest_f_classif'
            }
            self.feature_selection_methods[f'selectkbest_f_classif_{k}'] = selected_features
            print(f"   SelectKBest-f_classif-{k}: {selected_features[:5]}...")

        # SelectKBest with mutual_info_classif
        print("\n[2] SelectKBest (mutual_info_classif)...")
        for k in [5, 10]:
            selector = SelectKBest(mutual_info_classif, k=k)
            X_selected = selector.fit_transform(self.X_scaled, self.y_encoded)
            selected_features = [self.feature_cols[i] for i in selector.get_support(indices=True)]
            self.reduction_results[f'selectkbest_mutual_info_{k}'] = {
                'X': X_selected,
                'n_components': k,
                'description': f'SelectKBest-mutual_info-{k}',
                'model': selector,
                'features': selected_features,
                'scores': selector.scores_.tolist(),
                'method': 'selectkbest_mutual_info'
            }
            self.feature_selection_methods[f'selectkbest_mutual_info_{k}'] = selected_features
            print(f"   SelectKBest-mutual_info-{k}: {selected_features[:5]}...")

        # SelectFromModel using RandomForest
        print("\n[3] SelectFromModel (RandomForest)...")
        rf_selector = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_selector.fit(self.X_scaled, self.y_encoded)
        for k in [5, 10]:
            selector = SelectFromModel(rf_selector, threshold='mean', prefit=True)
            selector.max_features = k
            X_selected = selector.transform(self.X_scaled)
            selected_features = [self.feature_cols[i] for i in selector.get_support(indices=True)]
            self.reduction_results[f'selectfrommodel_rf_{k}'] = {
                'X': X_selected,
                'n_components': k,
                'description': f'SelectFromModel-RF-{k}',
                'model': selector,
                'features': selected_features,
                'scores': rf_selector.feature_importances_.tolist(),
                'method': 'selectfrommodel_rf'
            }
            self.feature_selection_methods[f'selectfrommodel_rf_{k}'] = selected_features
            print(f"   SelectFromModel-RF-{k}: {selected_features[:5]}...")

        # SelectFromModel using XGBoost
        if XGBOOST_AVAILABLE:
            print("\n[4] SelectFromModel (XGBoost)...")
            xgb_selector = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            xgb_selector.fit(self.X_scaled, self.y_encoded)
            for k in [5, 10]:
                selector = SelectFromModel(xgb_selector, threshold='mean', prefit=True)
                selector.max_features = k
                X_selected = selector.transform(self.X_scaled)
                selected_features = [self.feature_cols[i] for i in selector.get_support(indices=True)]
                self.reduction_results[f'selectfrommodel_xgb_{k}'] = {
                    'X': X_selected,
                    'n_components': k,
                    'description': f'SelectFromModel-XGB-{k}',
                    'model': selector,
                    'features': selected_features,
                    'scores': xgb_selector.feature_importances_.tolist(),
                    'method': 'selectfrommodel_xgb'
                }
                self.feature_selection_methods[f'selectfrommodel_xgb_{k}'] = selected_features
                print(f"   SelectFromModel-XGB-{k}: {selected_features[:5]}...")

        # No reduction (all features)
        print("\n[5] No Reduction (All Features)...")
        self.reduction_results['none'] = {
            'X': self.X_scaled,
            'n_components': self.X_scaled.shape[1],
            'description': 'No Reduction',
            'method': 'none'
        }

        # PCA reduction with original feature names
        print("\n[6] PCA Reduction (with Original Feature Names)...")
        for n_comp in [5, 10]:
            pca = PCA(n_components=n_comp, random_state=42)
            X_pca = pca.fit_transform(self.X_scaled)
            var_ratio = sum(pca.explained_variance_ratio_) * 100

            # Create original feature names for PCA components
            pca_feature_names = []
            for i in range(n_comp):
                loadings = np.abs(pca.components_[i])
                top_indices = np.argsort(loadings)[::-1][:3]
                top_features = [self.feature_cols[idx] for idx in top_indices]
                pca_feature_names.append(f'PC{i+1}_{top_features[0][:12]}')

            self.reduction_results[f'pca_{n_comp}'] = {
                'X': X_pca,
                'n_components': n_comp,
                'explained_var': var_ratio,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'description': f'PCA-{n_comp}',
                'model': pca,
                'method': 'pca',
                'features': pca_feature_names,
                'original_feature_names': self.feature_cols,
                'pca_loadings': pca.components_.tolist()
            }
            print(f"   PCA-{n_comp}: {var_ratio:.1f}% variance")
            print(f"      Feature names: {pca_feature_names}")


        # Save feature selection summary
        fs_summary = []
        for name, features in self.feature_selection_methods.items():
            fs_summary.append({
                'Method': name,
                'N_Features': len(features),
                'Features': ', '.join(features[:5]) + ('...' if len(features) > 5 else '')
            })
        pd.DataFrame(fs_summary).to_csv(
            os.path.join(OUTPUT_DIR, 'tables', 'feature_selection_methods.csv'), index=False)
        pd.DataFrame(fs_summary).to_excel(
            os.path.join(OUTPUT_DIR, 'tables', 'feature_selection_methods.xlsx'), index=False)

        print(f"\nTotal feature selection methods: {len(self.reduction_results)}")

        return self

    def enumerate_models(self):
        """Enumerate all model combinations with LOOCV"""
        print("\n" + "=" * 80)
        print("STEP 4: ENUMERATIVE MODELING WITH LOOCV (Two-Class)")
        print("=" * 80)

        models = {
            'DT_default': DecisionTreeClassifier(random_state=42),
            'DT_pruned': DecisionTreeClassifier(max_depth=5, random_state=42),
            'DT_entropy': DecisionTreeClassifier(criterion='entropy', random_state=42),
            'RF_default': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'RF_tuned': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            'RF_extratrees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'GB_default': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'GB_tuned': GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
            'HistGB_default': HistGradientBoostingClassifier(random_state=42),
            'HistGB_tuned': HistGradientBoostingClassifier(max_iter=200, max_depth=10, learning_rate=0.1, random_state=42),
            'XGB_default': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss') if XGBOOST_AVAILABLE else None,
            'XGB_tuned': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='mlogloss') if XGBOOST_AVAILABLE else None,
            'LGBM_default': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1) if LIGHTGBM_AVAILABLE else None,
            'LGBM_tuned': LGBMClassifier(n_estimators=200, max_depth=10, learning_rate=0.1, random_state=42, verbose=-1) if LIGHTGBM_AVAILABLE else None,
            'CatBoost_default': CatBoostClassifier(iterations=100, random_seed=42, verbose=0) if CATBOOST_AVAILABLE else None,
            'CatBoost_tuned': CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1, random_seed=42, verbose=0) if CATBOOST_AVAILABLE else None,
            'SVM_rbf': SVC(kernel='rbf', probability=True, random_state=42),
            'SVM_linear': SVC(kernel='linear', probability=True, random_state=42),
            'SVM_rbf_tuned': SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
            'KNN_3': KNeighborsClassifier(n_neighbors=3),
            'KNN_5': KNeighborsClassifier(n_neighbors=5),
            'KNN_7': KNeighborsClassifier(n_neighbors=7),
            'KNN_weighted': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'LR_default': LogisticRegression(max_iter=1000, random_state=42),
            'LR_l1': LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42),
            'LR_l2': LogisticRegression(penalty='l2', C=0.5, max_iter=2000, random_state=42),
            'LDA_default': LinearDiscriminantAnalysis(),
            'LDA_solver_svd': LinearDiscriminantAnalysis(solver='svd'),
            'LDA_solver_lsqr': LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
            'QDA_default': QuadraticDiscriminantAnalysis(),
            'QDA_reg': QuadraticDiscriminantAnalysis(reg_param=0.1),
            'NB_gaussian': GaussianNB(),
            'NB_bernoulli': BernoulliNB(),
            'MLP_small': MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42),
            'MLP_medium': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'Ridge_default': RidgeClassifier(random_state=42),
            'Ridge_alpha1': RidgeClassifier(alpha=1.0, random_state=42),
        }

        models = {k: v for k, v in models.items() if v is not None}

        cv_5fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        total_combinations = len(self.reduction_results) * len(models)
        current = 0

        print(f"\nTotal model combinations: {total_combinations}")
        print(f"Models available: {len(models)}")
        print(f"Feature selection methods: {len(self.reduction_results)}")
        print(f"Classification type: Two-class (Low, High)")
        print("-" * 80)

        for red_name, red_data in self.reduction_results.items():
            X_data = red_data['X']

            for model_name, model in models.items():
                current += 1

                try:
                    # 5-Fold CV
                    y_pred_cv = cross_val_predict(model, X_data, self.y_encoded, cv=cv_5fold)

                    cv_acc = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='accuracy')
                    cv_f1_macro = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='f1_macro')
                    cv_f1_weighted = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='f1_weighted')
                    cv_precision_macro = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='precision_macro')
                    cv_recall_macro = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='recall_macro')

                    try:
                        cv_roc_auc = cross_val_score(model, X_data, self.y_encoded, cv=cv_5fold, scoring='roc_auc')
                        roc_auc_mean = cv_roc_auc.mean()
                    except:
                        roc_auc_mean = 0.0

                    # Per-fold metrics
                    acc_per_fold = []
                    f1_per_fold = []

                    for train_idx, val_idx in cv_5fold.split(X_data, self.y_encoded):
                        X_train, X_val = X_data[train_idx], X_data[val_idx]
                        y_train, y_val = self.y_encoded[train_idx], self.y_encoded[val_idx]

                        from sklearn.base import clone
                        model_clone = clone(model)
                        model_clone.fit(X_train, y_train)
                        y_pred = model_clone.predict(X_val)

                        acc_per_fold.append(accuracy_score(y_val, y_pred))
                        f1_per_fold.append(f1_score(y_val, y_pred, average='macro'))

                    result = {
                        'reduction_method': red_name,
                        'reduction_description': red_data['description'],
                        'n_components': red_data['n_components'],
                        'model': model_name,
                        'model_class': model.__class__.__name__,
                        'features': red_data.get('features', []),
                        'accuracy_mean': cv_acc.mean(),
                        'accuracy_std': cv_acc.std(),
                        'f1_macro_mean': cv_f1_macro.mean(),
                        'f1_macro_std': cv_f1_macro.std(),
                        'f1_weighted_mean': cv_f1_weighted.mean(),
                        'precision_macro_mean': cv_precision_macro.mean(),
                        'recall_macro_mean': cv_recall_macro.mean(),
                        'roc_auc_mean': roc_auc_mean,
                        'accuracy_cv1': acc_per_fold[0], 'accuracy_cv2': acc_per_fold[1],
                        'accuracy_cv3': acc_per_fold[2], 'accuracy_cv4': acc_per_fold[3],
                        'accuracy_cv5': acc_per_fold[4],
                        'f1_cv1': f1_per_fold[0], 'f1_cv2': f1_per_fold[1],
                        'f1_cv3': f1_per_fold[2], 'f1_cv4': f1_per_fold[3],
                        'f1_cv5': f1_per_fold[4],
                        'X_data': X_data,
                        'model_template': model
                    }

                    self.all_results.append(result)

                    if current % 50 == 0 or current == total_combinations:
                        print(f"Progress: {current}/{total_combinations} ({(current/total_combinations*100):.1f}%)")

                except Exception as e:
                    continue

        print(f"\nCompleted! Evaluated {len(self.all_results)} combinations")

        # Save all results
        results_df = pd.DataFrame([{
            'Rank': i+1,
            'Reduction_Method': r['reduction_method'],
            'Reduction_Description': r['reduction_description'],
            'Model': r['model'],
            'Model_Class': r['model_class'],
            'N_Components': r['n_components'],
            'Accuracy_Mean': round(r['accuracy_mean'], 4),
            'Accuracy_Std': round(r['accuracy_std'], 4),
            'F1_Macro': round(r['f1_macro_mean'], 4),
            'F1_Weighted': round(r['f1_weighted_mean'], 4),
            'Precision_Macro': round(r['precision_macro_mean'], 4),
            'Recall_Macro': round(r['recall_macro_mean'], 4),
            'ROC_AUC': round(r['roc_auc_mean'], 4),
            'CV1': round(r['accuracy_cv1'], 4),
            'CV2': round(r['accuracy_cv2'], 4),
            'CV3': round(r['accuracy_cv3'], 4),
            'CV4': round(r['accuracy_cv4'], 4),
            'CV5': round(r['accuracy_cv5'], 4)
        } for i, r in enumerate(self.all_results)])

        results_df = results_df.sort_values('Accuracy_Mean', ascending=False)
        results_df['Rank'] = range(1, len(results_df) + 1)
        # Ensure tables directory exists
        os.makedirs(os.path.join(OUTPUT_DIR, 'tables'), exist_ok=True)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'all_model_results.csv'), index=False)
        results_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'all_model_results.xlsx'), index=False)

        return self

    def loocv_validation(self):
        """Leave-One-Out Cross-Validation"""
        print("\n" + "=" * 80)
        print("STEP 5: LOOCV (LEAVE-ONE-OUT CROSS-VALIDATION) - Two-Class")
        print("=" * 80)

        top_models = sorted(self.all_results, key=lambda x: x['accuracy_mean'], reverse=True)[:20]

        print(f"\nRunning LOOCV on top {len(top_models)} models...")
        print(f"Classification: Two-class (Low, High)")

        loocv = LeaveOneOut()

        for result in top_models:
            model_name = result['model']
            red_method = result['reduction_method']
            X_data = result['X_data']
            model_template = result['model_template']

            y_true_all = []
            y_pred_all = []
            y_proba_all = []

            for train_idx, val_idx in loocv.split(X_data):
                X_train, X_val = X_data[train_idx], X_data[val_idx]
                y_train, y_val = self.y_encoded[train_idx], self.y_encoded[val_idx]

                try:
                    from sklearn.base import clone
                    model = clone(model_template)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    y_proba = None
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val)

                    y_true_all.append(y_val[0])
                    y_pred_all.append(y_pred[0])
                    if y_proba is not None:
                        y_proba_all.append(y_proba[0])
                except Exception as e:
                    continue

            loocv_acc = accuracy_score(y_true_all, y_pred_all)
            loocv_f1 = f1_score(y_true_all, y_pred_all, average='macro')

            loocv_auc = 0.0
            if len(y_proba_all) > 0:
                y_proba_array = np.array(y_proba_all)
                if y_proba_array.shape[1] >= 2:
                    try:
                        loocv_auc = roc_auc_score(y_true_all, y_proba_array[:, 1])
                    except:
                        loocv_auc = 0.0

            loocv_result = {
                'model': model_name,
                'reduction_method': red_method,
                'reduction_description': result['reduction_description'],
                'n_components': result['n_components'],
                'cv5_accuracy': result['accuracy_mean'],
                'loocv_accuracy': loocv_acc,
                'cv5_f1': result['f1_macro_mean'],
                'loocv_f1': loocv_f1,
                'cv5_auc': result['roc_auc_mean'],
                'loocv_auc': loocv_auc,
                'n_samples': len(y_true_all),
                'classification_type': 'two-class'
            }
            self.loocv_results.append(loocv_result)

            print(f"  {model_name} ({red_method}): "
                  f"5-CV Acc={result['accuracy_mean']:.4f}, LOOCV Acc={loocv_acc:.4f}, "
                  f"5-CV F1={result['f1_macro_mean']:.4f}, LOOCV F1={loocv_f1:.4f}")

        loocv_df = pd.DataFrame(self.loocv_results)
        loocv_df = loocv_df.sort_values('loocv_accuracy', ascending=False)
        loocv_df.to_csv(os.path.join(OUTPUT_DIR, 'loocv_results', 'loocv_summary.csv'), index=False)
        loocv_df.to_excel(os.path.join(OUTPUT_DIR, 'loocv_results', 'loocv_summary.xlsx'), index=False)

        # Generate LOOCV comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        ax1 = axes[0]
        models = [r['model'][:15] for r in self.loocv_results]
        cv5_acc = [r['cv5_accuracy'] for r in self.loocv_results]
        loocv_acc = [r['loocv_accuracy'] for r in self.loocv_results]
        x = np.arange(len(models))
        width = 0.35
        ax1.barh(x - width/2, cv5_acc, width, label='5-Fold CV', color='steelblue')
        ax1.barh(x + width/2, loocv_acc, width, label='LOOCV', color='darkorange')
        ax1.set_yticks(x)
        ax1.set_yticklabels(models, fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('5-Fold CV vs LOOCV Accuracy (Two-Class)')
        ax1.legend()
        ax1.invert_yaxis()

        ax2 = axes[1]
        cv5_f1 = [r['cv5_f1'] for r in self.loocv_results]
        loocv_f1 = [r['loocv_f1'] for r in self.loocv_results]
        ax2.barh(x - width/2, cv5_f1, width, label='5-Fold CV', color='steelblue')
        ax2.barh(x + width/2, loocv_f1, width, label='LOOCV', color='darkorange')
        ax2.set_yticks(x)
        ax2.set_yticklabels(models, fontsize=8)
        ax2.set_xlabel('F1 Score (Macro)')
        ax2.set_title('5-Fold CV vs LOOCV F1 Score (Two-Class)')
        ax2.legend()
        ax2.invert_yaxis()

        ax3 = axes[2]
        cv5_auc = [r['cv5_auc'] for r in self.loocv_results]
        loocv_auc = [r['loocv_auc'] for r in self.loocv_results]
        ax3.barh(x - width/2, cv5_auc, width, label='5-Fold CV', color='steelblue')
        ax3.barh(x + width/2, loocv_auc, width, label='LOOCV', color='darkorange')
        ax3.set_yticks(x)
        ax3.set_yticklabels(models, fontsize=8)
        ax3.set_xlabel('ROC-AUC')
        ax3.set_title('5-Fold CV vs LOOCV ROC-AUC (Two-Class)')
        ax3.legend()
        ax3.invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'loocv_results', 'loocv_comparison.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, 'loocv_results', 'loocv_comparison.pdf'))
        plt.close()

        print(f"\nLOOCV results saved to: {OUTPUT_DIR}/loocv_results/")

        return self

    def find_best_model(self):
        """Find and analyze best models"""
        print("\n" + "=" * 80)
        print("STEP 6: BEST MODEL SELECTION (Two-Class)")
        print("=" * 80)

        self.all_results.sort(key=lambda x: (
            x['accuracy_mean'],
            x['f1_macro_mean'],
            x['roc_auc_mean']
        ), reverse=True)

        self.best_result = self.all_results[0]
        self.best_model_name = f"{self.best_result['reduction_description']} + {self.best_result['model']}"

        print("\n[TOP 20 MODEL COMBINATIONS - Two-Class Classification]")
        print("-" * 120)
        header = f"{'Rank':<5} {'Reduction':<25} {'Model':<20} {'Acc':<10} {'F1':<10} {'ROC-AUC':<10}"
        print(header)
        print("-" * 120)

        for i, result in enumerate(self.all_results[:20], 1):
            print(f"{i:<5} {result['reduction_method']:<25} {result['model']:<20} "
                  f"{result['accuracy_mean']:.4f}    {result['f1_macro_mean']:.4f}    {result['roc_auc_mean']:.4f}")

        print("\n" + "=" * 80)
        print(f"[BEST MODEL]")
        print(f"  Classification Type: Two-Class (Low, High)")
        print(f"  Reduction: {self.best_result['reduction_description']}")
        print(f"  Model: {self.best_result['model']}")
        print(f"  Accuracy: {self.best_result['accuracy_mean']:.4f} ± {self.best_result['accuracy_std']:.4f}")
        print(f"  F1 (Macro): {self.best_result['f1_macro_mean']:.4f}")
        print(f"  ROC-AUC: {self.best_result['roc_auc_mean']:.4f}")
        if self.best_result.get('features'):
            print(f"  Selected Features: {self.best_result['features'][:5]}...")
        print("=" * 80)

        return self

    def train_and_evaluate_best_model(self):
        """Train best model with comprehensive evaluation"""
        print("\n" + "=" * 80)
        print("STEP 7: TRAIN AND EVALUATE BEST MODEL (Two-Class)")
        print("=" * 80)

        X_data = self.best_result['X_data']

        X_train, X_test, y_train, y_test = train_test_split(
            X_data, self.y_encoded, test_size=0.2, random_state=42, stratify=self.y_encoded
        )

        model_template = self.best_result['model_template']
        try:
            from sklearn.base import clone
            self.final_model = clone(model_template)
        except:
            self.final_model = model_template
        self.final_model.fit(X_train, y_train)

        y_pred = self.final_model.predict(X_test)
        y_pred_proba = None
        if hasattr(self.final_model, 'predict_proba'):
            y_pred_proba = self.final_model.predict_proba(X_test)

        print("\n[TEST SET EVALUATION - Two-Class]")

        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1_macro = f1_score(y_test, y_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
        test_mcc = matthews_corrcoef(y_test, y_pred)
        test_kappa = cohen_kappa_score(y_test, y_pred)

        if y_pred_proba is not None:
            try:
                test_roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                test_roc_auc = 0.0
            try:
                test_log_loss = log_loss(y_test, y_pred_proba)
            except:
                test_log_loss = 0.0
            try:
                test_pr_auc = average_precision_score(y_test, y_pred_proba[:, 1])
            except:
                test_pr_auc = 0.0
        else:
            test_roc_auc = 0.0
            test_log_loss = 0.0
            test_pr_auc = 0.0

        print(f"  Classification: Two-class (Low, High)")
        print(f"  Accuracy: {test_accuracy:.4f}")
        print(f"  F1 (Macro): {test_f1_macro:.4f}")
        print(f"  F1 (Weighted): {test_f1_weighted:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  ROC-AUC: {test_roc_auc:.4f}")
        print(f"  Balanced Accuracy: {test_balanced_acc:.4f}")
        print(f"  MCC: {test_mcc:.4f}")
        print(f"  Kappa: {test_kappa:.4f}")
        print(f"  Log Loss: {test_log_loss:.4f}")
        print(f"  PR-AUC: {test_pr_auc:.4f}")

        print(f"\n[CLASSIFICATION REPORT - Two Classes]")
        report = classification_report(y_test, y_pred, target_names=self.class_names, output_dict=True)
        print(classification_report(y_test, y_pred, target_names=self.class_names))

        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        self.report = report
        self.cm = cm
        self.cm_df = cm_df
        self.cm_normalized = cm_normalized
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        self.test_accuracy = test_accuracy
        self.test_f1_macro = test_f1_macro
        self.test_roc_auc = test_roc_auc
        self.test_balanced_acc = test_balanced_acc
        self.test_mcc = test_mcc
        self.test_kappa = test_kappa
        self.test_log_loss = test_log_loss
        self.test_pr_auc = test_pr_auc

        # Get feature names for the current reduction
        red_method = self.best_result['reduction_method']
        reduction_info = self.reduction_results.get(red_method, {})
        if 'features' in reduction_info:
            self.feature_names = reduction_info['features']
        elif red_method == 'none':
            self.feature_names = self.feature_cols
        elif red_method.startswith('pca_'):
            n_comp = reduction_info['n_components']
            self.feature_names = [f'PC{i+1}' for i in range(n_comp)]
        else:
            self.feature_names = [f'F{i+1}' for i in range(X_data.shape[1])]

        if len(self.feature_names) != X_data.shape[1]:
            self.feature_names = [f'Feature_{i+1}' for i in range(X_data.shape[1])]

        metrics_data = []
        for cls in self.class_names:
            if cls in report:
                metrics_data.append({
                    'Category': cls,
                    'Threshold': {'Low': '<=3', 'High': '>3'}.get(cls, ''),
                    'Precision': round(report[cls]['precision'], 4),
                    'Recall': round(report[cls]['recall'], 4),
                    'F1_Score': round(report[cls]['f1-score'], 4),
                    'Support': int(report[cls]['support'])
                })
        metrics_data.extend([
            {'Category': 'Weighted Avg',
             'Threshold': '',
             'Precision': round(report['weighted avg']['precision'], 4),
             'Recall': round(report['weighted avg']['recall'], 4),
             'F1_Score': round(report['weighted avg']['f1-score'], 4),
             'Support': int(report['weighted avg']['support'])},
            {'Category': 'Macro Avg',
             'Threshold': '',
             'Precision': round(report['macro avg']['precision'], 4),
             'Recall': round(report['macro avg']['recall'], 4),
             'F1_Score': round(report['macro avg']['f1-score'], 4),
             'Support': int(report['macro avg']['support'])}
        ])

        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'classification_metrics.csv'), index=False)
        metrics_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'classification_metrics.xlsx'), index=False)

        cm_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'confusion_matrix.csv'))
        cm_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'confusion_matrix.xlsx'))

        self.model_info = {
            'model_class': model_template.__class__.__name__,
            'model_params': model_template.get_params() if hasattr(model_template, 'get_params') else {},
            'reduction_method': self.best_result['reduction_method'],
            'description': self.best_model_name,
            'cv_accuracy': self.best_result['accuracy_mean'],
            'cv_f1_macro': self.best_result['f1_macro_mean'],
            'cv_roc_auc': self.best_result['roc_auc_mean'],
            'test_accuracy': test_accuracy,
            'test_f1_macro': test_f1_macro,
            'test_roc_auc': test_roc_auc,
            'test_balanced_acc': test_balanced_acc,
            'test_mcc': test_mcc,
            'test_kappa': test_kappa,
            'n_all_features': len(self.feature_cols),
            'all_feature_names': self.feature_cols,
            'n_selected_features': len(self.feature_names),
            'selected_features': self.feature_names if hasattr(self, 'feature_names') else [],
            'n_samples': len(self.df),
            'class_names': self.class_names,
            'classification_type': 'two-class',
            'thresholds': {
                'Low': '<=3',
                'High': '>3'
            },
            'created_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        return self

    def generate_best_model_visualizations(self):
        """Generate comprehensive visualizations for the best model"""
        print("\n" + "=" * 80)
        print("STEP 8: GENERATE BEST MODEL VISUALIZATIONS (Two-Class)")
        print("=" * 80)

        plot_dir = os.path.join(OUTPUT_DIR, 'best_model_plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Confusion Matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax1 = axes[0]
        sns.heatmap(self.cm_df, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   annot_kws={'size': 14})
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_title('Confusion Matrix (Counts)\nTwo-Class: Low, High', fontsize=14)

        ax2 = axes[1]
        sns.heatmap(self.cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   annot_kws={'size': 14})
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_title('Confusion Matrix (Normalized)\nTwo-Class: Low, High', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, '01_confusion_matrix.png'), dpi=300)
        plt.savefig(os.path.join(plot_dir, '01_confusion_matrix.pdf'))
        plt.close()

        # ROC Curves
        if self.y_pred_proba is not None and self.y_pred_proba.shape[1] >= 2:
            n_classes = self.y_pred_proba.shape[1]

            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            colors = plt.cm.Set1(np.linspace(0, 1, n_classes))
            class_colors = ['#2ecc71', '#e74c3c'][:n_classes]

            for i, cls in enumerate(self.class_names):
                ax = axes[i] if i < len(axes) else axes[-1]
                y_test_binary = (self.y_test == i).astype(int)
                if len(np.unique(y_test_binary)) > 1:
                    fpr, tpr, _ = roc_curve(y_test_binary, self.y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, color=class_colors[i], lw=2,
                            label=f'{cls} (AUC = {roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], 'k--', lw=1)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC: {cls}')
                ax.legend(loc='lower right')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, '02_roc_curves.png'), dpi=300)
            plt.savefig(os.path.join(plot_dir, '02_roc_curves.pdf'))
            plt.close()

        # Feature Importance
        if hasattr(self.final_model, 'feature_importances_'):
            importances = self.final_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]

            fig, ax = plt.subplots(figsize=(12, 8))
            n_show = min(20, len(indices))
            y_pos = np.arange(n_show)
            sorted_importances = importances[indices[:n_show]]
            sorted_names = [self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}' for i in indices[:n_show]]

            colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_show))
            ax.barh(y_pos, sorted_importances[::-1], color=colors[::-1])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_names[::-1])
            ax.set_xlabel('Importance Score')
            ax.set_title(f'Top {n_show} Feature Importance (Two-Class Model)')

            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, '03_feature_importance.png'), dpi=300)
            plt.savefig(os.path.join(plot_dir, '03_feature_importance.pdf'))
            plt.close()

            importance_df = pd.DataFrame({
                'Feature': self.feature_names if len(self.feature_names) == len(importances) else [f'F{i}' for i in range(len(importances))],
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            importance_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'feature_importance.csv'), index=False)
            importance_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'feature_importance.xlsx'), index=False)

        print(f"\nGenerated visualizations in: {plot_dir}")

        return self

    def shap_analysis(self):
        """SHAP Analysis - Enhanced for SCI Publication Standards with Comprehensive Visualizations"""
        print("\n" + "=" * 80)
        print("STEP 9: SHAP ANALYSIS - SCI PUBLICATION STANDARDS (Two-Class)")
        print("=" * 80)

        if not SHAP_AVAILABLE:
            print("SHAP library not installed. Skipping SHAP analysis.")
            return self

        if not hasattr(self, 'final_model') or self.final_model is None:
            print("Best model not trained yet. Skipping SHAP analysis.")
            return self

        shap_dir = os.path.join(OUTPUT_DIR, 'shap_plots')
        sci_shap_dir = os.path.join(OUTPUT_DIR, 'sci_shap_analysis')
        os.makedirs(shap_dir, exist_ok=True)
        os.makedirs(sci_shap_dir, exist_ok=True)

        model = self.final_model
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        print(f"\nModel type: {model.__class__.__name__}")
        print(f"Number of features: {len(self.feature_names)}")

        try:
            # Determine model type and compute SHAP values
            if ('RandomForest' in model.__class__.__name__ or
                'GradientBoosting' in model.__class__.__name__ or
                'XGB' in model.__class__.__name__ or
                'LGBM' in model.__class__.__name__ or
                'CatBoost' in model.__class__.__name__ or
                'ExtraTrees' in model.__class__.__name__ or
                'DecisionTree' in model.__class__.__name__):

                print("Using TreeExplainer (tree-based model)...")
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_df)
                explainer_type = 'tree'

                if isinstance(shap_values, list):
                    for i, cls in enumerate(self.class_names):
                        plt.figure(figsize=(14, 10))
                        shap.summary_plot(shap_values[i], X_test_df, feature_names=self.feature_names,
                                        show=False, title=f'SHAP Summary - {cls}')
                        plt.tight_layout()
                        plt.savefig(os.path.join(shap_dir, f'shap_summary_{cls}.png'), dpi=300, bbox_inches='tight')
                        plt.close()

                    plt.figure(figsize=(14, 10))
                    shap.summary_plot(shap_values, X_test_df, feature_names=self.feature_names, show=False)
                    plt.tight_layout()
                    plt.savefig(os.path.join(shap_dir, 'shap_summary_plot.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    shap_values_combined = np.mean(np.abs(shap_values), axis=0)
                else:
                    shap_values_combined = shap_values

            elif 'LinearDiscriminantAnalysis' in model.__class__.__name__:
                print("Using LDA Coefficient-based SHAP explanation...")
                coefficients = model.coef_
                n_features = X_test_df.shape[1]
                n_samples = len(X_test_df)
                shap_values_combined = np.zeros((n_samples, n_features))

                if len(coefficients.shape) == 2:
                    for cls_idx in range(coefficients.shape[0]):
                        coef = coefficients[cls_idx]
                        for i in range(min(n_features, len(coef))):
                            shap_values_combined[:, i] += coef[i] * X_test_df.values[:, i]
                else:
                    coef_flat = coefficients.flatten()
                    for i in range(min(n_features, len(coef_flat))):
                        shap_values_combined[:, i] = coef_flat[i] * X_test_df.values[:, i]

                class LDADummyExplainer:
                    def __init__(self, model, X_train):
                        self.model = model
                        self.X_train = X_train
                        try:
                            self.expected_value = np.mean(model.decision_function(X_train))
                        except:
                            self.expected_value = 0.0

                explainer = LDADummyExplainer(model, self.X_train)
                explainer_type = 'lda'

            elif hasattr(model, 'predict_proba'):
                print("Using KernelExplainer (general model)...")
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                background_df = pd.DataFrame(background, columns=self.feature_names)
                explainer = shap.KernelExplainer(lambda x: model.predict_proba(x), background_df)
                shap_values = explainer.shap_values(X_test_df, nsamples=500)

                if isinstance(shap_values, list):
                    shap_values_combined = np.mean(np.abs(shap_values), axis=0)
                else:
                    shap_values_combined = shap_values
                explainer_type = 'kernel'
            else:
                print("Model does not support SHAP analysis.")
                return self

            # Process shap values
            if len(shap_values_combined.shape) == 1:
                shap_values_combined = shap_values_combined.reshape(-1, 1)

            shap_values_2d = shap_values_combined
            if len(shap_values_2d.shape) == 3:
                shap_values_2d = shap_values_2d[:, :, 1]

            # =====================================================================
            # SCI PUBLICATION-READY SHAP VISUALIZATIONS
            # =====================================================================
            print("\n" + "-" * 60)
            print("Generating SCI Publication-Ready SHAP Visualizations...")
            print("-" * 60)

            # 1. Beeswarm Plot (Publication Quality)
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values_2d, X_test_df, feature_names=self.feature_names, show=False)
            plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '01_shap_beeswarm.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '01_shap_beeswarm.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Beeswarm plot")

            # 2. Bar Plot (Publication Quality)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_2d, X_test_df, feature_names=self.feature_names, plot_type='bar', show=False)
            plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '02_shap_bar.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '02_shap_bar.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Bar plot")

            # 3. SHAP Dependence Plots for Top Features
            mean_abs_shap = np.abs(shap_values_2d).mean(axis=0)
            top_indices = np.argsort(mean_abs_shap)[::-1][:min(6, len(mean_abs_shap))]

            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            for idx, ax in zip(top_indices, axes.flatten()):
                feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f'F{idx}'
                shap.dependence_plot(idx, shap_values_2d, X_test_df, feature_names=self.feature_names, show=False, ax=ax)
                ax.set_title(f'{feat_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '03_shap_dependence.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '03_shap_dependence.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Dependence plots")

            # 4. Violin Plot
            plt.figure(figsize=(14, 10))
            shap.summary_plot(shap_values_2d, X_test_df, feature_names=self.feature_names, plot_type='violin', show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '04_shap_violin.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '04_shap_violin.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Violin plot")

            # 5. Heatmap Plot
            plt.figure(figsize=(16, 12))
            shap_subset = shap_values_2d[:min(50, len(shap_values_2d)), :]
            feat_subset = self.feature_names[:shap_subset.shape[1]]
            plt.imshow(shap_subset.T, aspect='auto', cmap='RdBu_r')
            plt.colorbar(label='SHAP Value', shrink=0.8)
            plt.xlabel('Samples', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.yticks(range(len(feat_subset)), feat_subset, fontsize=9)
            plt.title('SHAP Values Heatmap (First 50 Samples)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '05_shap_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '05_shap_heatmap.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Heatmap")

            # 6. Feature Importance with Error Bars
            mean_shap = np.abs(shap_values_2d).mean(axis=0)
            std_shap = np.abs(shap_values_2d).std(axis=0)
            feat_names = self.feature_names[:len(mean_shap)] if len(self.feature_names) >= len(mean_shap) else self.feature_names
            sorted_idx = np.argsort(mean_shap)[::-1]
            sorted_feat = [feat_names[i] for i in sorted_idx]
            sorted_mean = mean_shap[sorted_idx]
            sorted_std = std_shap[sorted_idx]

            plt.figure(figsize=(12, max(8, len(sorted_feat) * 0.4)))
            y_pos = np.arange(len(sorted_feat))
            plt.barh(y_pos, sorted_mean, xerr=sorted_std, color='steelblue', capsize=3, alpha=0.8)
            plt.yticks(y_pos, sorted_feat, fontsize=10)
            plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
            plt.title('SHAP Feature Importance with Std Dev', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(sci_shap_dir, '06_shap_importance_std.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(sci_shap_dir, '06_shap_importance_std.pdf'), bbox_inches='tight')
            plt.close()
            print("  [OK] Feature importance with error bars")

            # =====================================================================
            # SCI STATISTICAL SUMMARY TABLES
            # =====================================================================
            print("\n" + "-" * 60)
            print("Generating SHAP Statistical Summary Tables...")
            print("-" * 60)

            # Table 1: SHAP Statistics
            shap_stats = pd.DataFrame({
                'Feature': feat_names,
                'Mean_|SHAP|': mean_shap,
                'Std_SHAP': std_shap,
                'Median_SHAP': np.median(shap_values_2d, axis=0),
                'Min_SHAP': np.min(shap_values_2d, axis=0),
                'Max_SHAP': np.max(shap_values_2d, axis=0)
            }).sort_values('Mean_|SHAP|', ascending=False).reset_index(drop=True)
            shap_stats.to_csv(os.path.join(sci_shap_dir, 'table1_shap_stats.csv'), index=False)
            shap_stats.to_excel(os.path.join(sci_shap_dir, 'table1_shap_stats.xlsx'), index=False)
            print("  [OK] SHAP statistics table")

            # Table 2: SHAP Ranking
            shap_ranking = pd.DataFrame({
                'Rank': range(1, len(sorted_feat) + 1),
                'Feature': sorted_feat,
                'Mean_Abs_SHAP': sorted_mean,
                'Std': sorted_std,
                'Contribution_Pct': sorted_mean / sorted_mean.sum() * 100,
                'Cumulative_Pct': np.cumsum(sorted_mean / sorted_mean.sum() * 100)
            })
            shap_ranking.to_csv(os.path.join(sci_shap_dir, 'table2_shap_ranking.csv'), index=False)
            shap_ranking.to_excel(os.path.join(sci_shap_dir, 'table2_shap_ranking.xlsx'), index=False)
            print("  [OK] SHAP ranking table")

            # Table 3: Sample-level SHAP Values
            shap_df = pd.DataFrame(shap_values_2d, columns=feat_names)
            shap_df['Sample_ID'] = range(len(shap_df))
            shap_df['Predicted'] = model.predict(X_test_df)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test_df)
                shap_df['Prob_High'] = proba[:, 1] if proba.shape[1] > 1 else proba.flatten()
            shap_df.to_csv(os.path.join(sci_shap_dir, 'table3_sample_shap.csv'), index=False)
            shap_df.to_excel(os.path.join(sci_shap_dir, 'table3_sample_shap.xlsx'), index=False)
            print("  [OK] Sample SHAP table")

            # Save SHAP values
            np.save(os.path.join(sci_shap_dir, 'shap_values.npy'), shap_values_2d)
            print(f"\nSHAP analysis completed. Results saved to:")
            print(f"  - {shap_dir}/")
            print(f"  - {sci_shap_dir}/")

        except Exception as e:
            print(f"SHAP analysis failed: {e}")
            import traceback
            traceback.print_exc()

        return self

    def generate_all_model_comparison_plots(self):
        """Generate comparison plots for all models"""
        print("\n" + "=" * 80)
        print("STEP 10: GENERATE MODEL COMPARISON PLOTS (Two-Class)")
        print("=" * 80)

        fig, axes = plt.subplots(2, 2, figsize=(18, 16))

        top_20 = self.all_results[:20]
        y_pos = np.arange(len(top_20))
        accuracies = [r['accuracy_mean'] for r in top_20]
        f1_scores = [r['f1_macro_mean'] for r in top_20]
        roc_aucs = [r['roc_auc_mean'] for r in top_20]
        stds = [r['accuracy_std'] for r in top_20]

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_20)))[::-1]

        ax1 = axes[0, 0]
        ax1.barh(y_pos, accuracies, xerr=stds, color=colors, capsize=3, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"{r['model'][:15]}" for r in top_20], fontsize=8)
        ax1.set_xlabel('Accuracy')
        ax1.set_title('Top 20 Models - Accuracy (Two-Class)')
        ax1.invert_yaxis()

        ax2 = axes[0, 1]
        ax2.barh(y_pos, f1_scores, color=colors, alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"{r['model'][:15]}" for r in top_20], fontsize=8)
        ax2.set_xlabel('F1 Score (Macro)')
        ax2.set_title('Top 20 Models - F1 Score (Two-Class)')
        ax2.invert_yaxis()

        ax3 = axes[1, 0]
        ax3.barh(y_pos, roc_aucs, color=colors, alpha=0.8)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{r['model'][:15]}" for r in top_20], fontsize=8)
        ax3.set_xlabel('ROC-AUC')
        ax3.set_title('Top 20 Models - ROC-AUC (Two-Class)')
        ax3.invert_yaxis()

        ax4 = axes[1, 1]
        all_accs = [r['accuracy_mean'] for r in self.all_results]
        all_f1s = [r['f1_macro_mean'] for r in self.all_results]
        all_aucs = [r['roc_auc_mean'] for r in self.all_results]
        scatter = ax4.scatter(all_accs, all_f1s, c=all_aucs, cmap='viridis', alpha=0.6, s=50, edgecolors='white')
        ax4.set_xlabel('Accuracy')
        ax4.set_ylabel('F1 Score')
        ax4.set_title('Accuracy vs F1 Score (colored by ROC-AUC) - Two-Class')
        plt.colorbar(scatter, ax=ax4, label='ROC-AUC')
        ax4.scatter([self.best_result['accuracy_mean']], [self.best_result['f1_macro_mean']],
                   c='red', s=200, marker='*', label='Best', zorder=5)
        ax4.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'model_comparison.png'), dpi=300)
        plt.savefig(os.path.join(OUTPUT_DIR, 'figures', 'model_comparison.pdf'))
        plt.close()

        print(f"\nGenerated comparison plots in: {OUTPUT_DIR}/figures/")

    # =========================================================================
    # NEW V10 SCI PUBLICATION ANALYSIS METHODS
    # =========================================================================

    def generate_sci_comprehensive_analysis(self):
        """NEW V10: Generate comprehensive SCI publication-ready analysis for best model"""
        print("\n" + "=" * 80)
        print("STEP 11: SCI PUBLICATION-READY ANALYSIS (V10 NEW FEATURE)")
        print("=" * 80)
        print("Generating comprehensive analysis for best model...")
        print(f"  Best Model: {self.best_model_name}")
        print(f"  Classification: Two-class (Low, High)")

        sci_dir = os.path.join(OUTPUT_DIR, 'sci_plots')
        sci_table_dir = os.path.join(OUTPUT_DIR, 'sci_tables')
        os.makedirs(sci_dir, exist_ok=True)
        os.makedirs(sci_table_dir, exist_ok=True)

        # 1. Generate comprehensive feature importance ranking
        self._generate_feature_importance_ranking()

        # 2. Generate permutation importance with statistical tests
        self._generate_permutation_importance_analysis()

        # 3. Generate calibration curves
        self._generate_calibration_curves()

        # 4. Generate learning curves
        self._generate_learning_curves()

        # 5. Generate decision boundary visualization (if 2D)
        self._generate_decision_boundary()

        # 6. Generate precision-recall curve
        self._generate_pr_curves()

        # 7. Generate cross-validation stability analysis
        self._generate_cv_stability_analysis()

        # 8. Generate comprehensive metrics table for SCI
        self._generate_sci_metrics_table()

        # 9. Generate feature correlation analysis for best model
        self._generate_feature_correlation_analysis()

        # 10. Generate comprehensive SCI report
        self._generate_sci_comprehensive_report()

        print(f"\nSCI analysis completed. Results saved to:")
        print(f"  - {sci_dir}/")
        print(f"  - {sci_table_dir}/")

        return self

    def _generate_feature_importance_ranking(self):
        """NEW V10: Generate comprehensive feature importance ranking using multiple methods"""
        print("\n  [SCI-1] Generating Feature Importance Ranking...")

        X_data = self.best_result['X_data']
        feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'F{i}' for i in range(X_data.shape[1])]

        importance_results = {}

        # Method 1: Built-in feature importance (if available)
        if hasattr(self.final_model, 'feature_importances_'):
            importance_results['builtin'] = {
                'importance': self.final_model.feature_importances_,
                'method': 'Built-in Feature Importance'
            }

        # Method 2: Mutual Information
        mi_scores = mutual_info_classif(X_data, self.y_encoded, random_state=42)
        importance_results['mutual_info'] = {
            'importance': mi_scores,
            'method': 'Mutual Information'
        }

        # Method 3: ANOVA F-value
        f_scores, _ = f_classif(X_data, self.y_encoded)
        importance_results['anova'] = {
            'importance': f_scores,
            'method': 'ANOVA F-value'
        }

        # Method 4: Random Forest importance (re-trained for consistency)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_data, self.y_encoded)
        importance_results['random_forest'] = {
            'importance': rf.feature_importances_,
            'method': 'Random Forest'
        }

        # Method 5: XGBoost importance (if available)
        if XGBOOST_AVAILABLE:
            xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
            xgb.fit(X_data, self.y_encoded)
            importance_results['xgboost'] = {
                'importance': xgb.feature_importances_,
                'method': 'XGBoost'
            }

        # Method 6: Logistic Regression coefficients (absolute values)
        if 'LogisticRegression' in self.final_model.__class__.__name__:
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_data, self.y_encoded)
            importance_results['logistic'] = {
                'importance': np.abs(lr.coef_).mean(axis=0),
                'method': 'Logistic Regression'
            }

        # Method 7: Correlation with target
        correlations = []
        for i in range(X_data.shape[1]):
            corr = np.abs(np.corrcoef(X_data[:, i], self.y_encoded)[0, 1])
            correlations.append(corr if not np.isnan(corr) else 0)
        importance_results['correlation'] = {
            'importance': np.array(correlations),
            'method': 'Correlation with Target'
        }

        # Store for later use
        self.sci_results['feature_importance'] = importance_results

        # Normalize and aggregate rankings
        n_features = X_data.shape[1]
        n_methods = len(importance_results)

        # Create ranking matrix
        ranking_matrix = np.zeros((n_features, n_methods))
        method_names = list(importance_results.keys())

        for j, method in enumerate(method_names):
            scores = importance_results[method]['importance']
            # Normalize to 0-1
            if scores.max() > scores.min():
                normalized = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                normalized = scores
            ranking_matrix[:, j] = normalized

        # Calculate mean importance across methods
        mean_importance = ranking_matrix.mean(axis=1)

        # Calculate rank for each method
        ranks = np.zeros((n_features, n_methods))
        for j in range(n_methods):
            ranks[:, j] = stats.rankdata(-importance_results[method_names[j]]['importance'])

        # Calculate mean rank
        mean_rank = ranks.mean(axis=1)

        # Create comprehensive DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names[:n_features],
            'Feature_ID': range(1, n_features + 1)
        })

        for method in method_names:
            importance_df[f'{method}_importance'] = importance_results[method]['importance'][:n_features]

        importance_df['Mean_Importance'] = mean_importance
        importance_df['Mean_Rank'] = mean_rank

        # Calculate rank for each method
        for j, method in enumerate(method_names):
            importance_df[f'{method}_rank'] = ranks[:, j]

        # Sort by mean importance
        importance_df = importance_df.sort_values('Mean_Importance', ascending=False)

        # Save to files
        importance_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_feature_importance_ranking.csv'), index=False)
        importance_df.to_excel(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_feature_importance_ranking.xlsx'), index=False)

        # Generate visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Plot 1: Top 20 features by mean importance (bar plot)
        ax1 = axes[0, 0]
        top_n = min(20, n_features)
        top_features = importance_df.head(top_n)
        y_pos = np.arange(top_n)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, top_n))
        ax1.barh(y_pos, top_features['Mean_Importance'].values[::-1], color=colors[::-1])
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_features['Feature'].values[::-1])
        ax1.set_xlabel('Normalized Mean Importance', fontsize=12)
        ax1.set_title('Top 20 Features by Mean Importance\n(Aggregated across multiple methods)', fontsize=12, fontweight='bold')
        ax1.invert_yaxis()

        # Plot 2: Feature importance comparison across methods (heatmap)
        ax2 = axes[0, 1]
        importance_matrix = np.array([importance_results[m]['importance'][:top_n] for m in method_names[:6]]).T
        # Normalize each column
        importance_matrix_norm = (importance_matrix - importance_matrix.min(axis=0)) / (importance_matrix.max(axis=0) - importance_matrix.min(axis=0) + 1e-10)
        im = ax2.imshow(importance_matrix_norm, cmap='YlOrRd', aspect='auto')
        ax2.set_xticks(range(min(6, len(method_names))))
        ax2.set_xticklabels([m.replace('_', '\n') for m in method_names[:6]], rotation=45, ha='right', fontsize=9)
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(importance_df['Feature'].values[:top_n], fontsize=9)
        ax2.set_title('Feature Importance Heatmap\n(Methods: MI, ANOVA, RF, XGB, Corr)', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax2, shrink=0.8)

        # Plot 3: Rank comparison boxplot
        ax3 = axes[1, 0]
        rank_data = importance_df[[f'{m}_rank' for m in method_names]].values
        bp = ax3.boxplot([importance_df[f'{m}_rank'].values for m in method_names],
                         labels=[m[:12] for m in method_names], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_ylabel('Rank', fontsize=12)
        ax3.set_title('Feature Ranking Distribution by Method', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)

        # Plot 4: Top 10 features - individual method comparison
        ax4 = axes[1, 1]
        top10_features = importance_df.head(10)['Feature'].values
        top10_idx = [list(feature_names).index(f) if f in feature_names else i for i, f in enumerate(top10_features)]

        x = np.arange(len(top10_features))
        width = 0.12
        colors_methods = plt.cm.Set2(np.linspace(0, 1, len(method_names)))

        for j, method in enumerate(method_names[:6]):  # Show max 6 methods
            vals = importance_results[method]['importance'][:n_features]
            if len(vals) >= len(top10_idx):
                top10_vals = vals[top10_idx[:len(top10_features)]]
                # Normalize
                if top10_vals.max() > top10_vals.min():
                    top10_vals_norm = (top10_vals - top10_vals.min()) / (top10_vals.max() - top10_vals.min())
                else:
                    top10_vals_norm = top10_vals
                ax4.bar(x + j * width - 2.5 * width, top10_vals_norm, width, label=method[:10], color=colors_methods[j])

        ax4.set_xticks(x)
        ax4.set_xticklabels([f[:15] for f in top10_features], rotation=45, ha='right', fontsize=9)
        ax4.set_ylabel('Normalized Importance', fontsize=12)
        ax4.set_title('Top 10 Features - Method Comparison', fontsize=12, fontweight='bold')
        ax4.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_01_feature_importance_ranking.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_01_feature_importance_ranking.pdf'))
        plt.close()

        # Save top 10 features list
        top10_rows = importance_df.head(10)
        top10_df = pd.DataFrame({
            'Rank': range(1, len(top10_rows) + 1),
            'Feature': top10_rows['Feature'].values,
            'Mean_Importance': top10_rows['Mean_Importance'].values,
            'Mean_Rank': top10_rows['Mean_Rank'].values
        })
        top10_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_top10_features.csv'), index=False)

        print(f"    Generated feature importance ranking for {n_features} features using {n_methods} methods")

        return self

    def _generate_permutation_importance_analysis(self):
        """NEW V10: Generate permutation importance with statistical significance tests"""
        print("\n  [SCI-2] Generating Permutation Importance Analysis...")

        X_data = self.best_result['X_data']
        feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'F{i}' for i in range(X_data.shape[1])]

        # Calculate permutation importance with multiple repeats
        n_repeats = 30  # 30 repeats for statistical significance

        print(f"    Calculating permutation importance ({n_repeats} repeats)...")

        perm_importance = permutation_importance(
            self.final_model, X_data, self.y_encoded,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1,
            scoring='accuracy'
        )

        # Calculate p-values using t-test against zero
        p_values = []
        for i in range(len(perm_importance.importances_mean)):
            mean = perm_importance.importances_mean[i]
            std = perm_importance.importances_std[i]
            # One-sample t-test against 0
            if std > 0:
                # Get individual permutation scores
                scores = perm_importance.importances[:, i]
                t_stat, p_val = stats.ttest_1samp(scores, 0)
                p_values.append(p_val)
            else:
                p_values.append(1.0)

        # Create DataFrame
        perm_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance_Mean': perm_importance.importances_mean,
            'Importance_Std': perm_importance.importances_std,
            'Importance_Mean_StdErr': perm_importance.importances_std / np.sqrt(n_repeats),
            'P_Value': p_values,
            'Significant_005': ['Yes' if p < 0.05 else 'No' for p in p_values],
            'Significant_001': ['Yes' if p < 0.01 else 'No' for p in p_values]
        })

        # Calculate confidence intervals
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        perm_df['CI_Lower'] = perm_df['Importance_Mean'] - z_score * perm_df['Importance_Std'] / np.sqrt(n_repeats)
        perm_df['CI_Upper'] = perm_df['Importance_Mean'] + z_score * perm_df['Importance_Std'] / np.sqrt(n_repeats)

        # Sort by importance
        perm_df = perm_df.sort_values('Importance_Mean', ascending=False)

        # Save to files
        perm_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_permutation_importance.csv'), index=False)
        perm_df.to_excel(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_permutation_importance.xlsx'), index=False)

        # Store for report
        self.sci_results['permutation_importance'] = {
            'results': perm_df,
            'n_repeats': n_repeats
        }

        # Generate visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 10))

        # Plot 1: Top 20 permutation importance with error bars
        ax1 = axes[0]
        top_n = min(20, len(perm_df))
        top_perm = perm_df.head(top_n)

        y_pos = np.arange(top_n)
        colors = ['green' if p < 0.05 else 'gray' for p in top_perm['P_Value'].values]

        ax1.barh(y_pos, top_perm['Importance_Mean'].values[::-1],
                xerr=top_perm['Importance_Std'].values[::-1],
                color=colors[::-1], capsize=3, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_perm['Feature'].values[::-1])
        ax1.set_xlabel('Permutation Importance (Accuracy Decrease)', fontsize=12)
        ax1.set_title(f'Permutation Importance with Statistical Significance\n(n={n_repeats} permutations, Green: p<0.05)', fontsize=12, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        ax1.invert_yaxis()

        # Plot 2: Importance vs P-value scatter
        ax2 = axes[1]
        scatter = ax2.scatter(perm_df['Importance_Mean'], -np.log10(perm_df['P_Value'] + 1e-10),
                             c=perm_df['Importance_Mean'], cmap='viridis', s=100, alpha=0.7)
        ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax2.set_xlabel('Permutation Importance', fontsize=12)
        ax2.set_ylabel('-log10(P-value)', fontsize=12)
        ax2.set_title('Statistical Significance of Feature Importance', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Importance')

        # Annotate top features
        for i, row in perm_df.head(5).iterrows():
            ax2.annotate(row['Feature'][:10], (row['Importance_Mean'], -np.log10(row['P_Value'] + 1e-10)),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax2.legend()

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_02_permutation_importance.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_02_permutation_importance.pdf'))
        plt.close()

        print(f"    Generated permutation importance with {n_repeats} repeats")
        print(f"    Significant features (p<0.05): {(perm_df['P_Value'] < 0.05).sum()}/{len(perm_df)}")

        return self

    def _generate_calibration_curves(self):
        """NEW V10: Generate calibration curves (reliability diagrams)"""
        print("\n  [SCI-3] Generating Calibration Curves...")

        if not hasattr(self.final_model, 'predict_proba'):
            print("    Model does not support probability predictions, skipping calibration curve")
            return

        X_data = self.best_result['X_data']
        y_true = self.y_encoded

        # Get probability predictions using cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_proba = cross_val_predict(self.final_model, X_data, y_true, cv=cv, method='predict_proba')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Calibration curve for each class
        ax1 = axes[0]
        for i, cls in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            prob_pos = y_proba[:, i]

            if len(np.unique(y_true_binary)) > 1:
                prob_true, prob_pred = calibration_curve(y_true_binary, prob_pos, n_bins=5, strategy='uniform')
                ax1.plot(prob_pred, prob_true, marker='o', label=f'{cls}', linewidth=2, markersize=8)

        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curves (Reliability Diagrams)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3)

        # Plot 2: Predicted probability distribution
        ax2 = axes[1]
        for i, cls in enumerate(self.class_names):
            mask = y_true == i
            ax2.hist(y_proba[mask, i], bins=20, alpha=0.5, label=f'{cls} (True)', density=True)

        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of Predicted Probabilities', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Confidence vs Accuracy
        ax3 = axes[2]
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []

        for i in range(len(confidence_bins) - 1):
            mask = (np.max(y_proba, axis=1) >= confidence_bins[i]) & (np.max(y_proba, axis=1) < confidence_bins[i+1])
            if mask.sum() > 0:
                y_pred_bin = np.argmax(y_proba[mask], axis=1)
                y_true_bin = y_true[mask]
                acc = accuracy_score(y_true_bin, y_pred_bin)
                bin_accuracies.append(acc)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(np.nan)
                bin_counts.append(0)

        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        ax3.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7, color='steelblue', label='Actual Accuracy')
        ax3.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=1.5)
        ax3.set_xlabel('Confidence (Max Probability)', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Confidence vs Accuracy', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.set_xlim([0, 1])
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)

        # Add count annotations
        for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
            ax3.annotate(f'n={count}', (center, bin_accuracies[i] + 0.02),
                        ha='center', fontsize=8, rotation=45)

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_03_calibration_curves.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_03_calibration_curves.pdf'))
        plt.close()

        print("    Generated calibration curves (reliability diagrams)")

        return self

    def _generate_learning_curves(self):
        """NEW V10: Generate learning curves"""
        print("\n  [SCI-4] Generating Learning Curves...")

        X_data = self.best_result['X_data']
        y_true = self.y_encoded

        # Calculate learning curves
        train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes_abs, train_scores, val_scores = learning_curve(
            self.final_model, X_data, y_true,
            train_sizes=train_sizes,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42
        )

        train_scores_mean = train_scores.mean(axis=1)
        train_scores_std = train_scores.std(axis=1)
        val_scores_mean = val_scores.mean(axis=1)
        val_scores_std = val_scores.std(axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Learning curve
        ax1 = axes[0]
        ax1.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.2, color='blue')
        ax1.fill_between(train_sizes_abs, val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std, alpha=0.2, color='orange')
        ax1.plot(train_sizes_abs, train_scores_mean, 'o-', color='blue', label='Training Score')
        ax1.plot(train_sizes_abs, val_scores_mean, 'o-', color='orange', label='Validation Score')
        ax1.set_xlabel('Training Set Size', fontsize=12)
        ax1.set_ylabel('Accuracy Score', fontsize=12)
        ax1.set_title('Learning Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # Add gap analysis
        gap = train_scores_mean - val_scores_mean
        ax1.set_title(f'Learning Curves (Avg Gap: {gap.mean():.3f})', fontsize=12, fontweight='bold')

        # Plot 2: Score differences
        ax2 = axes[1]
        ax2.bar(train_sizes_abs, gap, color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_xlabel('Training Set Size', fontsize=12)
        ax2.set_ylabel('Train - Validation Score', fontsize=12)
        ax2.set_title('Generalization Gap by Training Size', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Ensure directory exists
        sci_plots_dir = os.path.join(OUTPUT_DIR, 'sci_plots')
        sci_tables_dir = os.path.join(OUTPUT_DIR, 'sci_tables')
        os.makedirs(sci_plots_dir, exist_ok=True)
        os.makedirs(sci_tables_dir, exist_ok=True)

        # Save with explicit forward slashes for cross-platform compatibility
        png_path = os.path.join(sci_plots_dir, 'sci_04_learning_curves.png').replace('\\', '/')
        pdf_path = os.path.join(sci_plots_dir, 'sci_04_learning_curves.pdf').replace('\\', '/')
        csv_path = os.path.join(sci_tables_dir, 'sci_learning_curve_data.csv').replace('\\', '/')

        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()

        # Save learning curve data
        lc_df = pd.DataFrame({
            'Train_Size': train_sizes_abs,
            'Train_Score_Mean': train_scores_mean,
            'Train_Score_Std': train_scores_std,
            'Validation_Score_Mean': val_scores_mean,
            'Validation_Score_Std': val_scores_std,
            'Gap': gap
        })
        lc_df.to_csv(csv_path, index=False)

        print(f"    Generated learning curves (final validation score: {val_scores_mean[-1]:.4f})")

        return self

    def _generate_decision_boundary(self):
        """NEW V10: Generate decision boundary visualization"""
        print("\n  [SCI-5] Generating Decision Boundary Visualization...")

        X_data = self.best_result['X_data']
        feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'F{i}' for i in range(X_data.shape[1])]

        # Only for 2D visualization - use top 2 important features
        if X_data.shape[1] < 2:
            print("    Not enough features for 2D decision boundary")
            return

        # Get top 2 features based on importance
        if hasattr(self.final_model, 'feature_importances_'):
            importance = self.final_model.feature_importances_
        else:
            importance = np.abs(self.final_model.coef_).flatten() if hasattr(self.final_model, 'coef_') else np.ones(X_data.shape[1])

        top_2_idx = np.argsort(importance)[-2:]

        # Create 2D dataset
        X_2d = X_data[:, top_2_idx]

        # Train model on 2D data
        from sklearn.base import clone
        model_2d = clone(self.final_model)
        model_2d.fit(X_2d, self.y_encoded)

        # Create meshgrid
        x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
        y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))

        # Predict on meshgrid
        Z = model_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if Z.shape[1] >= 2:
            Z = Z[:, 1]  # Use probability of positive class

        Z = Z.reshape(xx.shape)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Define colors for each class
        class_colors = ['#2ecc71', '#e74c3c']  # Green for Low, Red for High

        # Plot 1: Decision boundary with data points
        ax1 = axes[0]
        contour = ax1.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.8)
        plt.colorbar(contour, ax=ax1)

        # Plot data points
        for i, cls in enumerate(self.class_names):
            mask = self.y_encoded == i
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=class_colors[i],
                       edgecolors='black', s=100, label=cls, marker='o' if i == 0 else 's')

        ax1.set_xlabel(f'{feature_names[top_2_idx[0]]}', fontsize=12)
        ax1.set_ylabel(f'{feature_names[top_2_idx[1]]}', fontsize=12)
        ax1.set_title(f'Decision Boundary\n({feature_names[top_2_idx[0]]} vs {feature_names[top_2_idx[1]]})', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')

        # Plot 2: Decision boundary with probability contours
        ax2 = axes[1]
        ax2.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['lightblue', 'lightgreen'], alpha=0.6)
        ax2.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2, linestyles='--')
        ax2.contour(xx, yy, Z, levels=[0.25, 0.75], colors='gray', linewidths=1, linestyles=':')

        for i, cls in enumerate(self.class_names):
            mask = self.y_encoded == i
            ax2.scatter(X_2d[mask, 0], X_2d[mask, 1], c=class_colors[i],
                       edgecolors='black', s=100, label=cls)

        ax2.set_xlabel(f'{feature_names[top_2_idx[0]]}', fontsize=12)
        ax2.set_ylabel(f'{feature_names[top_2_idx[1]]}', fontsize=12)
        ax2.set_title('Decision Regions with Probability Contours\n(Dashed line: 50% threshold)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_05_decision_boundary.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_05_decision_boundary.pdf'))
        plt.close()

        print(f"    Generated decision boundary (top features: {feature_names[top_2_idx[0]]}, {feature_names[top_2_idx[1]]})")

        return self

    def _generate_pr_curves(self):
        """NEW V10: Generate Precision-Recall curves"""
        print("\n  [SCI-6] Generating Precision-Recall Curves...")

        X_data = self.best_result['X_data']
        y_true = self.y_encoded

        # Get probability predictions using cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_proba = cross_val_predict(self.final_model, X_data, y_true, cv=cv, method='predict_proba')

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: PR curve for each class
        ax1 = axes[0]
        precision = {}
        recall = {}
        pr_auc = {}

        for i, cls in enumerate(self.class_names):
            y_true_binary = (y_true == i).astype(int)
            prob_pos = y_proba[:, i]

            if len(np.unique(y_true_binary)) > 1:
                prec, rec, _ = precision_recall_curve(y_true_binary, prob_pos)
                ap = average_precision_score(y_true_binary, prob_pos)
                precision[cls] = prec
                recall[cls] = rec
                pr_auc[cls] = ap

                ax1.plot(rec, prec, marker='o', label=f'{cls} (AP={ap:.3f})', linewidth=2, markersize=6)

        ax1.set_xlabel('Recall', fontsize=12)
        ax1.set_ylabel('Precision', fontsize=12)
        ax1.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1.05])

        # Plot 2: PR curve for binary classification (class 1 vs class 0)
        ax2 = axes[1]

        # Calculate baseline (random classifier)
        baseline = y_true.sum() / len(y_true)
        ax2.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline (Prevalence={baseline:.3f})')

        # Plot the main PR curve (class 1)
        y_true_binary = (y_true == 1).astype(int)
        prob_pos = y_proba[:, 1]

        prec, rec, thresholds = precision_recall_curve(y_true_binary, prob_pos)
        ap = average_precision_score(y_true_binary, prob_pos)

        ax2.fill_between(rec, prec, alpha=0.3, color='steelblue')
        ax2.plot(rec, prec, 'b-', linewidth=2, label=f'AP={ap:.3f}')
        ax2.scatter([rec[0]], [prec[0]], marker='o', s=100, color='blue', zorder=5, label='Threshold=0')
        ax2.scatter([rec[-1]], [prec[-1]], marker='s', s=100, color='blue', zorder=5, label='Threshold=1')

        ax2.set_xlabel('Recall', fontsize=12)
        ax2.set_ylabel('Precision', fontsize=12)
        ax2.set_title(f'Precision-Recall Curve (High Class)\nAP={ap:.4f}', fontsize=12, fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 1])
        ax2.set_ylim([0, 1.05])

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_06_precision_recall_curves.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_06_precision_recall_curves.pdf'))
        plt.close()

        # Save PR data - ensure all arrays have same length
        # precision_recall_curve returns (precision, recall, thresholds)
        # where thresholds has one less element than precision/recall
        min_len = min(len(rec), len(prec))
        rec_trimmed = rec[:min_len]
        prec_trimmed = prec[:min_len]

        pr_df = pd.DataFrame({
            'Recall': rec_trimmed,
            'Precision': prec_trimmed
        })
        pr_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_pr_curve_data.csv'), index=False)

        print(f"    Generated Precision-Recall curves (Average Precision: {ap:.4f})")

        return self

    def _generate_cv_stability_analysis(self):
        """NEW V10: Generate cross-validation stability analysis"""
        print("\n  [SCI-7] Generating Cross-Validation Stability Analysis...")

        X_data = self.best_result['X_data']
        y_true = self.y_encoded

        # Run multiple cross-validation runs with different random states
        n_runs = 10
        cv_results = {
            'accuracy': [],
            'f1_macro': [],
            'f1_weighted': [],
            'roc_auc': [],
            'precision_macro': [],
            'recall_macro': []
        }

        for seed in range(n_runs):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            acc = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='accuracy')
            f1_mac = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='f1_macro')
            f1_weight = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='f1_weighted')
            try:
                roc_auc = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='roc_auc')
            except:
                roc_auc = [0.5]
            prec_mac = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='precision_macro')
            rec_mac = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='recall_macro')

            cv_results['accuracy'].extend(acc)
            cv_results['f1_macro'].extend(f1_mac)
            cv_results['f1_weighted'].extend(f1_weight)
            cv_results['roc_auc'].extend(roc_auc)
            cv_results['precision_macro'].extend(prec_mac)
            cv_results['recall_macro'].extend(rec_mac)

        # Calculate statistics
        cv_stats = {}
        for metric, values in cv_results.items():
            cv_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'iqr_25': np.percentile(values, 25),
                'iqr_75': np.percentile(values, 75),
                'cv': np.std(values) / np.mean(values) * 100  # Coefficient of variation
            }

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Box plots of CV results across runs
        ax1 = axes[0, 0]
        metrics = ['accuracy', 'f1_macro', 'roc_auc']
        data_to_plot = [cv_results[m] for m in metrics]
        bp = ax1.boxplot(data_to_plot, labels=[m.replace('_', '\n').title() for m in metrics], patch_artist=True)
        colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Cross-Validation Score Distribution\n(10 runs, 5-fold each)', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Score variability across runs
        ax2 = axes[0, 1]
        run_means = []
        run_stds = []
        for seed in range(n_runs):
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            acc = cross_val_score(self.final_model, X_data, y_true, cv=cv, scoring='accuracy')
            run_means.append(np.mean(acc))
            run_stds.append(np.std(acc))

        x = range(n_runs)
        ax2.errorbar(x, run_means, yerr=run_stds, fmt='o-', capsize=5, color='steelblue', markersize=8)
        ax2.axhline(y=np.mean(run_means), color='red', linestyle='--', label=f'Overall Mean: {np.mean(run_means):.4f}')
        ax2.fill_between(x, np.array(run_means) - np.array(run_stds),
                       np.array(run_means) + np.array(run_stds), alpha=0.2, color='steelblue')
        ax2.set_xlabel('Cross-Validation Run', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Accuracy Stability Across CV Runs', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Violin plots
        ax3 = axes[1, 0]
        data_for_violin = [cv_results[m] for m in ['accuracy', 'f1_macro', 'roc_auc']]
        vp = ax3.violinplot(data_for_violin, positions=range(len(metrics)), showmeans=True, showmedians=True)
        ax3.set_xticks(range(len(metrics)))
        ax3.set_xticklabels([m.replace('_', '\n').title() for m in metrics])
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_title('Score Distribution (Violin Plot)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Coefficient of Variation comparison
        ax4 = axes[1, 1]
        metrics_all = list(cv_stats.keys())
        cv_values = [cv_stats[m]['cv'] for m in metrics_all]
        colors = plt.cm.RdYlGn_r(np.array(cv_values) / max(cv_values))

        bars = ax4.bar(range(len(metrics_all)), cv_values, color=colors)
        ax4.set_xticks(range(len(metrics_all)))
        ax4.set_xticklabels([m.replace('_', '\n').title() for m in metrics_all], rotation=45, ha='right')
        ax4.set_ylabel('Coefficient of Variation (%)', fontsize=12)
        ax4.set_title('Model Stability (Lower CV = More Stable)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, cv_val in zip(bars, cv_values):
            ax4.annotate(f'{cv_val:.2f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_07_cv_stability.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_07_cv_stability.pdf'))
        plt.close()

        # Save CV stability data
        cv_stats_df = pd.DataFrame(cv_stats).T
        cv_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_cv_stability_stats.csv'))

        print(f"    Generated CV stability analysis ({n_runs} runs)")
        print(f"    Mean Accuracy: {cv_stats['accuracy']['mean']:.4f} ± {cv_stats['accuracy']['std']:.4f}")
        print(f"    Coefficient of Variation: {cv_stats['accuracy']['cv']:.2f}%")

        return self

    def _generate_sci_metrics_table(self):
        """NEW V10: Generate comprehensive SCI publication metrics table"""
        print("\n  [SCI-8] Generating SCI Metrics Table...")

        X_data = self.best_result['X_data']
        y_true = self.y_encoded

        # Calculate comprehensive metrics using cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        metrics_dict = {
            'Accuracy': [],
            'Balanced Accuracy': [],
            'F1 Score (Macro)': [],
            'F1 Score (Weighted)': [],
            'Precision (Macro)': [],
            'Recall (Macro)': [],
            'ROC-AUC': [],
            'PR-AUC': [],
            'Log Loss': [],
            'MCC': [],
            'Cohen\'s Kappa': []
        }

        for train_idx, val_idx in cv.split(X_data, y_true):
            X_train, X_val = X_data[train_idx], X_data[val_idx]
            y_train, y_val = y_true[train_idx], y_true[val_idx]

            from sklearn.base import clone
            model = clone(self.final_model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_val)
                try:
                    roc_auc = roc_auc_score(y_val, y_proba[:, 1])
                    pr_auc = average_precision_score(y_val, y_proba[:, 1])
                    ll = log_loss(y_val, y_proba)
                except:
                    roc_auc = 0.5
                    pr_auc = 0
                    ll = 0
            else:
                roc_auc = 0.5
                pr_auc = 0
                ll = 0

            metrics_dict['Accuracy'].append(accuracy_score(y_val, y_pred))
            metrics_dict['Balanced Accuracy'].append(balanced_accuracy_score(y_val, y_pred))
            metrics_dict['F1 Score (Macro)'].append(f1_score(y_val, y_pred, average='macro'))
            metrics_dict['F1 Score (Weighted)'].append(f1_score(y_val, y_pred, average='weighted'))
            metrics_dict['Precision (Macro)'].append(precision_score(y_val, y_pred, average='macro', zero_division=0))
            metrics_dict['Recall (Macro)'].append(recall_score(y_val, y_pred, average='macro', zero_division=0))
            metrics_dict['ROC-AUC'].append(roc_auc)
            metrics_dict['PR-AUC'].append(pr_auc)
            metrics_dict['Log Loss'].append(ll)
            metrics_dict['MCC'].append(matthews_corrcoef(y_val, y_pred))
            metrics_dict['Cohen\'s Kappa'].append(cohen_kappa_score(y_val, y_pred))

        # Calculate statistics
        sci_metrics_df = pd.DataFrame({
            'Metric': list(metrics_dict.keys()),
            'Mean': [np.mean(v) for v in metrics_dict.values()],
            'Std': [np.std(v) for v in metrics_dict.values()],
            'Median': [np.median(v) for v in metrics_dict.values()],
            'Min': [np.min(v) for v in metrics_dict.values()],
            'Max': [np.max(v) for v in metrics_dict.values()],
            '95% CI Lower': [np.percentile(v, 2.5) for v in metrics_dict.values()],
            '95% CI Upper': [np.percentile(v, 97.5) for v in metrics_dict.values()]
        })

        sci_metrics_df['Formatted'] = sci_metrics_df.apply(
            lambda x: f"{x['Mean']:.4f} ± {x['Std']:.4f}", axis=1
        )

        # Save to files
        sci_metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_comprehensive_metrics.csv'), index=False)
        sci_metrics_df.to_excel(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_comprehensive_metrics.xlsx'), index=False)

        # Create per-class metrics
        per_class_metrics = []
        for cls in self.class_names:
            class_metrics = {
                'Class': cls,
                'Threshold': {'Low': '<=3', 'High': '>3'}.get(cls, ''),
            }

            class_metrics_list = {k: [] for k in ['Precision', 'Recall', 'F1 Score']}

            for train_idx, val_idx in cv.split(X_data, y_true):
                X_train, X_val = X_data[train_idx], X_data[val_idx]
                y_train, y_val = y_true[train_idx], y_true[val_idx]

                from sklearn.base import clone
                model = clone(self.final_model)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                report = classification_report(y_val, y_pred, target_names=self.class_names, output_dict=True, zero_division=0)

                class_metrics_list['Precision'].append(report[cls]['precision'])
                class_metrics_list['Recall'].append(report[cls]['recall'])
                class_metrics_list['F1 Score'].append(report[cls]['f1-score'])

            for metric, values in class_metrics_list.items():
                class_metrics[f'{metric}_Mean'] = np.mean(values)
                class_metrics[f'{metric}_Std'] = np.std(values)

            per_class_metrics.append(class_metrics)

        per_class_df = pd.DataFrame(per_class_metrics)
        per_class_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_per_class_metrics.csv'), index=False)
        per_class_df.to_excel(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_per_class_metrics.xlsx'), index=False)

        # Store for report
        self.sci_results['metrics'] = {
            'overall': sci_metrics_df,
            'per_class': per_class_df
        }

        print(f"    Generated comprehensive SCI metrics table")
        print(f"    Metrics calculated: {len(metrics_dict)}")
        for _, row in sci_metrics_df.iterrows():
            print(f"      {row['Metric']}: {row['Formatted']}")

        return self

    def _generate_feature_correlation_analysis(self):
        """NEW V10: Generate feature correlation analysis for selected features"""
        print("\n  [SCI-9] Generating Feature Correlation Analysis...")

        X_data = self.best_result['X_data']
        feature_names = self.feature_names if hasattr(self, 'feature_names') else [f'F{i}' for i in range(X_data.shape[1])]

        # Create correlation matrix for selected features
        X_df = pd.DataFrame(X_data, columns=feature_names)
        corr_matrix = X_df.corr()

        # Generate visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))

        # Plot 1: Correlation heatmap
        ax1 = axes[0]
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, ax=ax1,
                   cbar_kws={"shrink": 0.8},
                   annot=False, fmt='.2f')
        ax1.set_title('Feature Correlation Matrix\n(Selected Features)', fontsize=12, fontweight='bold')

        # Plot 2: Feature-target correlation
        ax2 = axes[1]
        target_corr = []
        for i in range(X_data.shape[1]):
            corr = np.corrcoef(X_data[:, i], self.y_encoded)[0, 1]
            target_corr.append(corr if not np.isnan(corr) else 0)

        target_corr = np.abs(target_corr)  # Use absolute correlation
        sorted_idx = np.argsort(target_corr)[::-1]
        n_show = min(20, len(target_corr))

        colors = plt.cm.viridis(target_corr[sorted_idx[:n_show]])
        ax2.barh(range(n_show), target_corr[sorted_idx[:n_show]][::-1], color=colors[::-1])
        ax2.set_yticks(range(n_show))
        ax2.set_yticklabels([feature_names[sorted_idx[i]] for i in range(n_show)][::-1])
        ax2.set_xlabel('Absolute Correlation with Target', fontsize=12)
        ax2.set_title('Feature-Target Correlation\n(Top 20 Features)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()

        plt.tight_layout()
        safe_savefig(plt.gcf(), os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_08_feature_correlation.png'),
                    os.path.join(OUTPUT_DIR, 'sci_plots', 'sci_08_feature_correlation.pdf'))
        plt.close()

        # Save correlation data
        corr_df = pd.DataFrame({
            'Feature': feature_names,
            'Target_Correlation': [np.corrcoef(X_data[:, i], self.y_encoded)[0, 1] for i in range(X_data.shape[1])],
            'Abs_Target_Correlation': [np.abs(np.corrcoef(X_data[:, i], self.y_encoded)[0, 1]) for i in range(X_data.shape[1])]
        })
        corr_df = corr_df.sort_values('Abs_Target_Correlation', ascending=False)
        corr_df.to_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_feature_target_correlation.csv'), index=False)

        print(f"    Generated feature correlation analysis for {len(feature_names)} features")

        return self

    def _generate_sci_comprehensive_report(self):
        """NEW V10: Generate comprehensive SCI publication report"""
        print("\n  [SCI-10] Generating SCI Comprehensive Report...")

        report_content = f"""# SCI Publication-Ready Analysis Report - V10
## Best Model: {self.best_model_name}
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Dataset Information

| Property | Value |
|----------|-------|
| Total Samples | {len(self.df)} |
| Total Features | {len(self.feature_cols)} |
| Selected Features | {len(self.feature_names)} |
| Classes | {', '.join(self.class_names)} |
| Classification Type | Two-class (Low ≤3, High >3) |

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
"""

        for cls in self.class_names:
            count = (self.y_encoded == list(self.class_names).index(cls)).sum()
            pct = count / len(self.y_encoded) * 100
            report_content += f"| {cls} | {count} | {pct:.1f}% |\n"

        report_content += f"""
---

## 2. Best Model Configuration

| Property | Value |
|----------|-------|
| Model | {self.best_result['model']} |
| Model Class | {self.best_result['model_class']} |
| Feature Selection | {self.best_result['reduction_description']} |
| Number of Features | {self.best_result['n_components']} |

### Selected Features ({len(self.feature_names)} features)
"""

        for i, feat in enumerate(self.feature_names, 1):
            report_content += f"{i}. {feat}\n"

        report_content += f"""
---

## 3. Cross-Validation Results (5-Fold Stratified CV)

### Overall Performance Metrics
| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
"""

        if 'metrics' in self.sci_results:
            for _, row in self.sci_results['metrics']['overall'].iterrows():
                report_content += f"| {row['Metric']} | {row['Formatted']} | [{row['95% CI Lower']:.4f}, {row['95% CI Upper']:.4f}] |\n"

        report_content += f"""
### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
"""

        if 'metrics' in self.sci_results:
            for _, row in self.sci_results['metrics']['per_class'].iterrows():
                prec_str = f"{row['Precision_Mean']:.4f} ± {row['Precision_Std']:.4f}"
                rec_str = f"{row['Recall_Mean']:.4f} ± {row['Recall_Std']:.4f}"
                f1_str = f"{row['F1 Score_Mean']:.4f} ± {row['F1 Score_Std']:.4f}"
                report_content += f"| {row['Class']} | {prec_str} | {rec_str} | {f1_str} |\n"

        report_content += f"""
---

## 4. Feature Importance Ranking

The feature importance was calculated using multiple methods:
1. Built-in feature importance
2. Mutual Information
3. ANOVA F-value
4. Random Forest importance
5. XGBoost importance (if available)
6. Correlation with target

### Top 10 Most Important Features
"""

        if 'feature_importance' in self.sci_results:
            top10_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_top10_features.csv'))
            for _, row in top10_df.iterrows():
                report_content += f"{int(row['Rank'])}. **{row['Feature']}** (Importance: {row['Mean_Importance']:.4f}, Rank: {row['Mean_Rank']:.1f})\n"

        report_content += f"""
---

## 5. Permutation Importance Analysis

Statistical significance testing was performed using {self.sci_results.get('permutation_importance', {}).get('n_repeats', 30)} permutation repeats.

"""

        if 'permutation_importance' in self.sci_results:
            perm_df = self.sci_results['permutation_importance']['results']
            sig_count = (perm_df['P_Value'] < 0.05).sum()
            report_content += f"**Significant Features (p<0.05):** {sig_count}/{len(perm_df)}\n\n"

            report_content += "| Feature | Importance | Std | P-value | Significant |\n"
            report_content += "|---------|------------|-----|--------|-------------|\n"

            for _, row in perm_df.head(10).iterrows():
                sig = '✓' if row['P_Value'] < 0.05 else '✗'
                report_content += f"| {row['Feature']} | {row['Importance_Mean']:.4f} | {row['Importance_Std']:.4f} | {row['P_Value']:.4e} | {sig} |\n"

        report_content += f"""
---

## 6. Generated Figures

### SCI Publication-Ready Figures (High Resolution, 300 DPI)

1. **sci_01_feature_importance_ranking.png/pdf**
   - Feature importance ranking using multiple methods
   - Comparison across different feature selection methods

2. **sci_02_permutation_importance.png/pdf**
   - Permutation importance with statistical significance
   - P-value analysis for each feature

3. **sci_03_calibration_curves.png/pdf**
   - Calibration curves (reliability diagrams)
   - Predicted probability distribution

4. **sci_04_learning_curves.png/pdf**
   - Learning curves showing training/validation performance
   - Generalization gap analysis

5. **sci_05_decision_boundary.png/pdf**
   - 2D decision boundary visualization
   - Decision regions with probability contours

6. **sci_06_precision_recall_curves.png/pdf**
   - Precision-Recall curves for each class
   - Average Precision (AP) scores

7. **sci_07_cv_stability.png/pdf**
   - Cross-validation stability analysis
   - Coefficient of variation analysis

8. **sci_08_feature_correlation.png/pdf**
   - Feature correlation matrix
   - Feature-target correlation analysis

---

## 7. Generated Tables

### SCI Tables (CSV and Excel format)

1. **sci_comprehensive_metrics.csv/xlsx** - Complete metrics table
2. **sci_per_class_metrics.csv/xlsx** - Per-class performance
3. **sci_feature_importance_ranking.csv/xlsx** - Feature importance by multiple methods
4. **sci_permutation_importance.csv/xlsx** - Permutation importance with p-values
5. **sci_top10_features.csv** - Top 10 features summary
6. **sci_feature_target_correlation.csv** - Feature-target correlations
7. **sci_cv_stability_stats.csv** - CV stability statistics
8. **sci_learning_curve_data.csv** - Learning curve data
9. **sci_pr_curve_data.csv** - Precision-Recall curve data

---

## 8. Model Interpretation

### Key Findings

1. **Best Performing Model**: {self.best_result['model']} with {self.best_result['reduction_description']}
2. **Expected Performance**: Accuracy = {self.best_result['accuracy_mean']:.4f}, F1 = {self.best_result['f1_macro_mean']:.4f}
3. **Most Important Features**: Top 3 features identified by multiple methods

### Clinical/Biological Interpretation

- Features with highest importance contribute most to {self.class_names[0]} vs {self.class_names[1]} classification
- The model can potentially be used to predict MIC category from peptide features

---

## 9. Statistical Methods Used

1. **Cross-Validation**: 5-Fold Stratified Cross-Validation
2. **Feature Importance**: Multiple methods (MI, ANOVA, RF, XGB, Correlation)
3. **Statistical Tests**: Permutation tests with t-test for significance
4. **Confidence Intervals**: 95% CI calculated from CV folds
5. **Stability Analysis**: {10} repeated CV runs

---

## 10. Reproducibility

- Random Seed: 42 (used throughout the analysis)
- Software Version: V10
- Date Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

**Report Generated by AMP Modeling V10 - SCI Publication Analysis Module**
"""

        with open(os.path.join(OUTPUT_DIR, 'sci_tables', 'sci_comprehensive_report.md'), 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"    Generated comprehensive SCI report")

        return self

    def save_model_and_report(self):
        """Save model and generate comprehensive report"""
        print("\n" + "=" * 80)
        print("STEP 12: SAVE MODEL AND GENERATE REPORT (V10 - Two-Class + SCI)")
        print("=" * 80)

        with open(os.path.join(OUTPUT_DIR, 'best_model.pkl'), 'wb') as f:
            pickle.dump(self.final_model, f)
        with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        with open(os.path.join(OUTPUT_DIR, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        if 'model' in self.reduction_results.get(self.best_result['reduction_method'], {}):
            with open(os.path.join(OUTPUT_DIR, 'feature_selector.pkl'), 'wb') as f:
                pickle.dump(self.reduction_results[self.best_result['reduction_method']]['model'], f)

        with open(os.path.join(OUTPUT_DIR, 'model_info.json'), 'w') as f:
            json.dump(self.model_info, f, indent=2, ensure_ascii=False)

        # Save selected features
        if hasattr(self, 'feature_names') and self.feature_names:
            selected_df = pd.DataFrame({
                'Rank': range(1, len(self.feature_names) + 1),
                'Feature': self.feature_names
            })
            selected_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'selected_features.csv'), index=False)
            selected_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'selected_features.xlsx'), index=False)

        # Save ALL original features
        all_features_df = pd.DataFrame({
            'Rank': range(1, len(self.feature_cols) + 1),
            'Feature': self.feature_cols,
            'DataType': [self.X_raw_df[col].dtype for col in self.feature_cols]
        })
        all_features_df.to_csv(os.path.join(OUTPUT_DIR, 'tables', 'all_original_features.csv'), index=False)
        all_features_df.to_excel(os.path.join(OUTPUT_DIR, 'tables', 'all_original_features.xlsx'), index=False)

        # Generate comprehensive markdown report
        report_content = f"""# AMP MIC Prediction Model Report - V10
## Enhanced Version with LOOCV, Feature Selection, LDA SHAP + SCI Analysis
## Classification Type: Two-Class (Low, High)

## Classification Thresholds
| Category | Exp_Log2_MIC Range | Description |
|----------|-------------------|-------------|
| **Low** | <= 3 | Sensitive (lower MIC indicates better activity) |
| **High** | > 3 | Less sensitive/Resistant (higher MIC indicates reduced activity) |

## Dataset Information
- **Number of samples**: {len(self.df)}
- **Number of features**: {len(self.feature_cols)}
- **Number of classes**: {self.n_classes} (Two-class)
- **Classes**: {', '.join(self.class_names)}

## Class Distribution
| Category | Threshold | Count | Percentage |
|----------|-----------|-------|------------|
"""
        for cls in self.class_names:
            count = self.df['MIC_Category'].value_counts().get(cls, 0)
            pct = count / len(self.df) * 100
            threshold = {'Low': '<=3', 'High': '>3'}.get(cls, '')
            report_content += f"| {cls} | {threshold} | {count} | {pct:.1f}% |\n"

        report_content += f"""
## Feature Selection Methods (V10)
- **SelectKBest (f_classif)**: 5, 10 features
- **SelectKBest (mutual_info)**: 5, 10 features
- **SelectFromModel (RandomForest)**: 5, 10 features
- **SelectFromModel (XGBoost)**: 5, 10 features
- **PCA**: 5, 10 components
- **No Reduction**: All features

## Best Model Configuration
- **Reduction Method**: {self.best_result['reduction_description']}
- **Model**: {self.best_result['model']}
- **Model Class**: {self.best_result['model_class']}
- **Number of Selected Features**: {self.best_result['n_components']}

## All Original Features (for Scaler): {len(self.feature_cols)}
{', '.join(self.feature_cols[:20])}
{"..." if len(self.feature_cols) > 20 else ""}

## Selected Features (for Model): {len(self.feature_names)}
{', '.join(self.feature_names)}

## Cross-Validation Results (5-Fold)
| Metric | Value |
|--------|-------|
| **Accuracy** | {self.best_result['accuracy_mean']:.4f} ± {self.best_result['accuracy_std']:.4f} |
| **F1 (Macro)** | {self.best_result['f1_macro_mean']:.4f} |
| **F1 (Weighted)** | {self.best_result['f1_weighted_mean']:.4f} |
| **Precision (Macro)** | {self.best_result['precision_macro_mean']:.4f} |
| **Recall (Macro)** | {self.best_result['recall_macro_mean']:.4f} |
| **ROC-AUC** | {self.best_result['roc_auc_mean']:.4f} |

## Test Set Results
| Metric | Value |
|--------|-------|
| **Accuracy** | {self.model_info['test_accuracy']:.4f} |
| **F1 (Macro)** | {self.model_info['test_f1_macro']:.4f} |
| **ROC-AUC** | {self.model_info['test_roc_auc']:.4f} |
| **Balanced Accuracy** | {self.model_info['test_balanced_acc']:.4f} |
| **MCC** | {self.model_info['test_mcc']:.4f} |
| **Kappa** | {self.model_info['test_kappa']:.4f} |

## NEW V10: SCI Publication Analysis Features
The following additional analysis has been generated for SCI publication:

### SCI Figures (in sci_plots/)
1. sci_01_feature_importance_ranking.png/pdf - Feature importance by multiple methods
2. sci_02_permutation_importance.png/pdf - Permutation importance with p-values
3. sci_03_calibration_curves.png/pdf - Calibration curves
4. sci_04_learning_curves.png/pdf - Learning curves
5. sci_05_decision_boundary.png/pdf - Decision boundary visualization
6. sci_06_precision_recall_curves.png/pdf - Precision-Recall curves
7. sci_07_cv_stability.png/pdf - CV stability analysis
8. sci_08_feature_correlation.png/pdf - Feature correlation analysis

### SCI Tables (in sci_tables/)
1. sci_comprehensive_metrics.csv/xlsx - Complete metrics
2. sci_per_class_metrics.csv/xlsx - Per-class performance
3. sci_feature_importance_ranking.csv/xlsx - Feature importance ranking
4. sci_permutation_importance.csv/xlsx - Permutation importance
5. sci_top10_features.csv - Top 10 features
6. sci_feature_target_correlation.csv - Feature-target correlations
7. sci_cv_stability_stats.csv - CV stability statistics
8. sci_comprehensive_report.md - Complete SCI report

## Generated Files

### Tables
- `tables/all_model_results.csv/xlsx` - Complete model comparison
- `tables/classification_metrics.csv/xlsx` - Classification metrics per class
- `tables/confusion_matrix.csv/xlsx` - Confusion matrix
- `tables/feature_importance.csv/xlsx` - Feature importance
- `tables/selected_features.csv/xlsx` - Selected features for best model
- `tables/all_original_features.csv/xlsx` - ALL original features (for prediction)
- `tables/feature_selection_methods.csv/xlsx` - Feature selection methods summary
- `tables/class_distribution.csv/xlsx` - Class distribution summary

### Model Files
- `best_model.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `label_encoder.pkl` - Label encoder
- `feature_selector.pkl` - Feature selector
- `model_info.json` - Model configuration

### Visualizations
- `figures/00_class_distribution.png/pdf` - Class distribution pie chart
- `figures/00_feature_distributions.png` - Feature distributions
- `figures/00_feature_boxplots.png` - Feature box plots
- `figures/00_correlation_heatmap.png/pdf` - Feature correlation
- `figures/00_pca_2d_projection.png` - PCA projection
- `figures/model_comparison.png/pdf` - Model comparison plots
- `best_model_plots/01_confusion_matrix.png/pdf` - Confusion matrices
- `best_model_plots/02_roc_curves.png/pdf` - ROC curves
- `best_model_plots/03_feature_importance.png/pdf` - Feature importance
- `shap_plots/*.png/pdf` - SHAP analysis plots
- `loocv_results/*.csv/xlsx` - LOOCV results
- `sci_plots/*.png/pdf` - SCI publication figures
- `sci_tables/*.csv/xlsx` - SCI publication tables

---
**Model Version**: V10 (Two-Class + SCI Analysis)
**Classification**: Low (<=3), High (>3)
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(os.path.join(OUTPUT_DIR, 'report.md'), 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n" + "=" * 80)
        print("V10 MODELING COMPLETE! (Two-Class Classification + SCI Analysis)")
        print("=" * 80)
        print(f"\nOutput Directory: {OUTPUT_DIR}")
        print(f"\nModels Evaluated: {len(self.all_results)}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.best_result['accuracy_mean']:.4f}")
        print(f"Best F1 (Macro): {self.best_result['f1_macro_mean']:.4f}")
        print(f"Best ROC-AUC: {self.best_result['roc_auc_mean']:.4f}")
        print(f"\nClassification Type: Two-class (Low, High)")
        print(f"\nNEW V10 SCI Features:")
        print(f"  - Feature importance ranking (7 methods)")
        print(f"  - Permutation importance with p-values")
        print(f"  - Calibration curves")
        print(f"  - Learning curves")
        print(f"  - Decision boundary visualization")
        print(f"  - Precision-Recall curves")
        print(f"  - CV stability analysis")
        print(f"  - Comprehensive SCI tables")


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("AMP ENUMERATIVE MODELING - V10")
    print("Two-Class Classification: Low (<=3), High (>3)")
    print("Enhanced with LOOCV, Feature Selection, LDA SHAP + SCI Analysis")
    print("=" * 80)
    print("\nFeatures:")
    print("  1. Multiple feature selection methods (SelectKBest, SelectFromModel)")
    print("  2. LOOCV validation for top 20 models")
    print("  3. LDA SHAP explanation")
    print("  4. Feature counts: 5 and 10 only")
    print("  5. Comprehensive model evaluation")
    print("  6. Two-class classification (Low, High)")
    print("  7. NEW: SCI Publication-Ready Analysis (V10)")
    print("     - Feature importance ranking (7 methods)")
    print("     - Permutation importance with p-values")
    print("     - Calibration curves")
    print("     - Learning curves")
    print("     - Decision boundary visualization")
    print("     - Precision-Recall curves")
    print("     - CV stability analysis")
    print("=" * 80 + "\n")

    # Try multiple possible data file locations
    possible_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Q1_19_Seq_Model_Data.xlsx'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'user_input_files', 'Q1_19_Seq_Model_Data.xlsx'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'user_input_files', 'Q1_19_Seq_Model_Data.xlsx'),
    ]

    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break

    if data_path is None:
        print(f"Error: Data file not found. Tried the following locations:")
        for path in possible_paths:
            print(f"  - {path}")
        return

    print(f"Loading data from: {data_path}")

    modeler = AMPModelingV10(data_path)

    modeler.preprocess()
    modeler.scale_features()
    modeler.feature_selection()
    modeler.enumerate_models()
    modeler.loocv_validation()
    modeler.find_best_model()
    modeler.train_and_evaluate_best_model()
    modeler.generate_best_model_visualizations()
    modeler.generate_all_model_comparison_plots()
    modeler.shap_analysis()
    # NEW V10: SCI comprehensive analysis
    modeler.generate_sci_comprehensive_analysis()
    modeler.save_model_and_report()

    print("\nAll results saved to 'V10_results/' folder")


if __name__ == "__main__":
    main()
