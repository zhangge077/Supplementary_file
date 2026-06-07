"""
Peptide Property Analysis System
For analyzing relationships between peptide properties and target variable (binary or continuous)
Author: [Your Name]
Date: [Date]
Description: This system analyzes the relationship between peptide properties and target activity.
MODIFIED: MIC group analysis changed from three-class (Low/Mid/High) to two-class (Low/High) with Mann-Whitney U test.
Low: Exp_Log2_MIC <= 3, High: Exp_Log2_MIC > 3.
"""

import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
# Set backend to Agg to avoid display
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import pointbiserialr, pearsonr, spearmanr, chi2_contingency
from scipy.stats import kruskal, mannwhitneyu
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, auc, precision_recall_curve, mean_squared_error,
                           mean_absolute_error, r2_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# Set display options
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PeptideAnalyzer:
    """
    Peptide Analyzer for analyzing relationships with target variable
    Supports both binary (0/1) and continuous targets
    """

    def __init__(self, file_path):
        """
        Initialize peptide analyzer
        Args:
            file_path: Path to Excel file
        """
        self.data = pd.read_excel(file_path)
        self.results = {}
        self.models = {}
        self.feature_importance = {}
        self.correlation_matrix = None
        self.target_type = None  # 'binary' or 'continuous'
        self.model_results_df = None  # 新增：存储模型结果
        print(f"Data loaded successfully. Shape: {self.data.shape}")

        # 自动新增 ΔHCS4 / ΔHCS3 特征列：
        # Delta_HCS4 = AMP_Pairs1_HCS4_Score - AMP_Pairs2_HCS4_Score
        # Delta_HCS3 = AMP_Pairs1_HCS3_Score - AMP_Pairs2_HCS3_Score
        self.add_delta_hcs_features(verbose=True)

        # Create output directory
        self.output_dir = "peptide_analysis_results"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        # Create subdirectories for plots
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        self.interval_dir = os.path.join(self.plot_dir, "interval_analysis")
        self.corr_dir = os.path.join(self.plot_dir, "correlations")
        self.model_dir = os.path.join(self.plot_dir, "model_results")
        self.feature_dir = os.path.join(self.plot_dir, "feature_importance")

        for dir_path in [self.interval_dir, self.corr_dir, self.model_dir, self.feature_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    # ==================== Data Preparation Methods ====================

    def _find_first_existing_col(self, candidates):
        """
        Return the first existing column name from candidates.
        """
        for c in candidates:
            if c in self.data.columns:
                return c
        return None

    def add_delta_hcs_features(self, verbose=False):
        """
        Add directional HCS difference features to self.data.

        Delta_HCS4 = AMP_Pairs1_HCS4_Score - AMP_Pairs2_HCS4_Score
        Delta_HCS3 = AMP_Pairs1_HCS3_Score - AMP_Pairs2_HCS3_Score

        These two features are then available for:
        1) column selection in the interactive menu;
        2) MIC-group summary output:
           Feature, MIC_Group, N, Mean, Median, Std, Q10, Q25, Q75, Q90, Min, Max.
        """
        p1_hcs4 = self._find_first_existing_col([
            "AMP_Pairs1_HCS4_Score", "Pairs1_HCS4_Score", "P1_HCS4", "Pairs1_HCS4"
        ])
        p2_hcs4 = self._find_first_existing_col([
            "AMP_Pairs2_HCS4_Score", "Pairs2_HCS4_Score", "P2_HCS4", "Pairs2_HCS4"
        ])
        p1_hcs3 = self._find_first_existing_col([
            "AMP_Pairs1_HCS3_Score", "Pairs1_HCS3_Score", "P1_HCS3", "Pairs1_HCS3"
        ])
        p2_hcs3 = self._find_first_existing_col([
            "AMP_Pairs2_HCS3_Score", "Pairs2_HCS3_Score", "P2_HCS3", "Pairs2_HCS3"
        ])

        created = []

        if p1_hcs4 and p2_hcs4:
            self.data[p1_hcs4] = pd.to_numeric(self.data[p1_hcs4], errors="coerce")
            self.data[p2_hcs4] = pd.to_numeric(self.data[p2_hcs4], errors="coerce")
            self.data["Delta_HCS4"] = self.data[p1_hcs4] - self.data[p2_hcs4]
            created.append("Delta_HCS4")
        else:
            if verbose:
                print("[WARN] Delta_HCS4 not created: missing Pairs1/Pairs2 HCS4 columns.")

        if p1_hcs3 and p2_hcs3:
            self.data[p1_hcs3] = pd.to_numeric(self.data[p1_hcs3], errors="coerce")
            self.data[p2_hcs3] = pd.to_numeric(self.data[p2_hcs3], errors="coerce")
            self.data["Delta_HCS3"] = self.data[p1_hcs3] - self.data[p2_hcs3]
            created.append("Delta_HCS3")
        else:
            if verbose:
                print("[WARN] Delta_HCS3 not created: missing Pairs1/Pairs2 HCS3 columns.")

        if verbose and created:
            print("[INFO] Added derived features:", ", ".join(created))
            if "Delta_HCS4" in created:
                print(f"       Delta_HCS4 = {p1_hcs4} - {p2_hcs4}")
            if "Delta_HCS3" in created:
                print(f"       Delta_HCS3 = {p1_hcs3} - {p2_hcs3}")

        return created

    def parse_column_selection(self, column_string):
        """
        Parse column selection string like "1,2-4,6"
        Returns list of column indices
        """
        selected_indices = set()

        # Split by comma
        parts = column_string.replace(' ', '').split(',')

        for part in parts:
            if '-' in part:
                # Range like 2-4
                start, end = map(int, part.split('-'))
                selected_indices.update(range(start-1, end))  # Convert to 0-based
            else:
                # Single number
                selected_indices.add(int(part) - 1)  # Convert to 0-based

        # Sort and return
        return sorted(list(selected_indices))

    def get_columns_by_selection(self):
        """
        Display columns and get user selection
        """
        print("\n" + "="*60)
        print("AVAILABLE COLUMNS IN YOUR DATA:")
        print("="*60)

        columns = list(self.data.columns)
        for i, col in enumerate(columns, 1):
            print(f"{i:3}. {col}")

        print("-"*60)

        # Get factor columns
        while True:
            factor_input = input("\nSelect factor columns (enter numbers/ranges like '1,2-4,6'): ").strip()
            try:
                factor_indices = self.parse_column_selection(factor_input)
                factor_cols = [columns[i] for i in factor_indices]

                # Validate
                if len(factor_cols) == 0:
                    print("Error: No columns selected. Please try again.")
                    continue

                print(f"\nSelected factor columns: {', '.join(factor_cols)}")
                break
            except Exception as e:
                print(f"Error parsing input: {e}. Please try again.")

        # Get target column
        while True:
            target_input = input("\nSelect target column (enter number): ").strip()
            try:
                target_idx = int(target_input) - 1
                if 0 <= target_idx < len(columns):
                    target_col = columns[target_idx]

                    # Determine target type
                    unique_values = self.data[target_col].dropna().nunique()
                    if unique_values == 2:
                        self.target_type = "binary"
                        print(f"\nSelected target column: {target_col} - BINARY target detected")
                    else:
                        self.target_type = "continuous"
                        print(f"\nSelected target column: {target_col} - CONTINUOUS target detected")
                    break
                else:
                    print(f"Error: Invalid column number. Please enter a number between 1 and {len(columns)}")
            except ValueError:
                print("Error: Please enter a valid number")

        return factor_cols, target_col

    # ==================== Basic Statistical Methods ====================

    def basic_descriptive_stats(self, factor_cols, target_col):

        """
        Basic descriptive statistics for both binary and continuous targets.
        Robust to object/string columns: numeric summaries are computed on coerced numeric values.
        """
        stats_dict = {}

        # Target column
        if target_col in self.data.columns:
            if self.target_type == "binary":
                target_data = self.data[target_col].dropna()
                target_num = pd.to_numeric(target_data, errors="coerce")
                n_positive = (target_num == 1).sum()
                n_negative = (target_num == 0).sum()
                total = len(target_data)

                stats_dict[target_col] = {
                    'Type': 'Binary Target',
                    'Positive (1)': f"{int(n_positive)} ({(n_positive/total*100 if total else 0):.1f}%)",
                    'Negative (0)': f"{int(n_negative)} ({(n_negative/total*100 if total else 0):.1f}%)",
                    'Total': total,
                    'Missing': int(self.data[target_col].isnull().sum())
                }
            else:
                target_num = pd.to_numeric(self.data[target_col], errors="coerce")
                stats_dict[target_col] = {
                    'Type': 'Continuous Target',
                    'Mean': float(target_num.mean()),
                    'Std': float(target_num.std()),
                    'Min': float(target_num.min()),
                    '25%': float(target_num.quantile(0.25)),
                    'Median': float(target_num.median()),
                    '75%': float(target_num.quantile(0.75)),
                    'Max': float(target_num.max()),
                    'Missing': int(self.data[target_col].isnull().sum())
                }

        # Factor columns
        for col in factor_cols:
            if col in self.data.columns:
                col_num = pd.to_numeric(self.data[col], errors="coerce")
                stats_dict[col] = {
                    'Type': 'Continuous Feature',
                    'Mean': float(col_num.mean()),
                    'Std': float(col_num.std()),
                    'Min': float(col_num.min()),
                    '25%': float(col_num.quantile(0.25)),
                    'Median': float(col_num.median()),
                    '75%': float(col_num.quantile(0.75)),
                    'Max': float(col_num.max()),
                    'Missing': int(self.data[col].isnull().sum())
                }

        return pd.DataFrame(stats_dict).T

    def calculate_correlations(self, factor_cols, target_col):
        """Calculate correlation matrix and feature-target correlations with robust numeric coercion."""
        cols = list(factor_cols) + [target_col]

        # Coerce all columns to numeric (non-numeric -> NaN), then drop NaNs to align rows
        corr_df = self.data[cols].copy()
        for c in cols:
            corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")

        corr_data = corr_df.dropna()

        # If too few rows, return NaNs
        if corr_data.shape[0] < 3:
            correlation_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
            self.correlation_matrix = correlation_matrix
            target_corr = correlation_matrix[target_col].drop(target_col)
            target_corr = target_corr.sort_values(key=lambda x: np.abs(x), ascending=False)
            p_values = {f: np.nan for f in factor_cols}
            return correlation_matrix, target_corr, p_values

        correlation_matrix = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

        def _safe_pearson(x, y):
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if x.size < 3 or np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
                return (np.nan, np.nan)
            return pearsonr(x, y)

        def _safe_pointbiserial(continuous, binary):
            continuous = np.asarray(continuous, dtype=float)
            binary = np.asarray(binary, dtype=float)
            if continuous.size < 3 or np.nanstd(continuous) < 1e-12 or np.nanstd(binary) < 1e-12:
                return (np.nan, np.nan)
            return pointbiserialr(continuous, binary)

        # Fill matrix
        for i in cols:
            for j in cols:
                if i == j:
                    correlation_matrix.loc[i, j] = 1.0
                    continue

                if i == target_col or j == target_col:
                    if self.target_type == "binary":
                        if i == target_col:
                            binary = corr_data[i].round().astype(int).values
                            cont = corr_data[j].values
                        else:
                            binary = corr_data[j].round().astype(int).values
                            cont = corr_data[i].values
                        r, _p = _safe_pointbiserial(cont, binary)
                    else:
                        r, _p = _safe_pearson(corr_data[i].values, corr_data[j].values)
                else:
                    r, _p = _safe_pearson(corr_data[i].values, corr_data[j].values)

                correlation_matrix.loc[i, j] = r
                correlation_matrix.loc[j, i] = r

        self.correlation_matrix = correlation_matrix

        # Target correlations
        target_corr = self.correlation_matrix[target_col].drop(target_col)
        target_corr = target_corr.sort_values(key=lambda x: np.abs(x), ascending=False)

        # p-values (pairwise dropna, aligned)
        p_values = {}
        for factor in factor_cols:
            pair = corr_df[[factor, target_col]].dropna()
            if pair.shape[0] < 3:
                p_values[factor] = np.nan
                continue
            x = pair[factor].values.astype(float)
            y = pair[target_col].values.astype(float)
            if self.target_type == "binary":
                r, p = _safe_pointbiserial(x, y)
            else:
                r, p = _safe_pearson(x, y)
            p_values[factor] = p

        return correlation_matrix, target_corr, p_values

    def _to_numeric_series(self, s):
        """Coerce a pandas Series to numeric (float), turning non-numeric to NaN."""
        return pd.to_numeric(s, errors="coerce")

    def _numeric_df(self, cols):
        """Return a DataFrame with selected columns coerced to numeric."""
        df = self.data[list(cols)].copy()
        for c in cols:
            df[c] = self._to_numeric_series(df[c])
        return df

    def _safe_polyfit_1d(self, x, y):
        """Safe 1D linear polyfit; returns (z, poly1d) or (None, None)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size < 3:
            return None, None
        if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
            return None, None
        z = np.polyfit(x, y, 1)
        return z, np.poly1d(z)

    def plot_correlation_single(self, factor, target_col, correlation_value, p_value):
        """Plot single correlation plot for one factor vs target (robust numeric cleaning)."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Coerce to numeric and align x/y
        df = self._numeric_df([factor, target_col]).dropna()

        # If no numeric data after cleaning, return an empty informative plot
        if df.shape[0] == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "No numeric data after cleaning", ha="center", va="center")
                ax.axis("off")
            plt.tight_layout()
            filename = f"{factor.replace(' ', '_').replace('/', '_')}_correlation.png"
            filepath = os.path.join(self.corr_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath

        if self.target_type == "binary":
            ax1 = axes[0]
            df[target_col] = df[target_col].round().astype(int)

            for label, color in [(0, 'blue'), (1, 'red')]:
                subset = df[df[target_col] == label]
                ax1.scatter(subset[factor].values, subset[target_col].values,
                            alpha=0.6, color=color, s=50,
                            label=f'Class {label} (n={len(subset)})')

            ax1.set_xlabel(factor, fontsize=12)
            ax1.set_ylabel('Target (0/1)', fontsize=12)
            ax1.set_title(
                f'{factor} vs Target\nPoint-biserial r = {correlation_value:.3f}, p = {p_value:.4f}',
                fontsize=14, fontweight='bold'
            )
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            box_data = [df[df[target_col] == 0][factor].values,
                        df[df[target_col] == 1][factor].values]
            box = ax2.boxplot(box_data, labels=['Class 0', 'Class 1'], patch_artist=True)

            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(box['boxes'], colors):
                patch.set_facecolor(color)

            ax2.set_ylabel(factor, fontsize=12)
            ax2.set_title('Distribution by Target Class', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        else:
            ax1 = axes[0]
            x = df[factor].astype(float).values
            y = df[target_col].astype(float).values

            ax1.scatter(x, y, alpha=0.6, s=50, c='steelblue')

            z, poly = self._safe_polyfit_1d(x, y)
            if poly is not None:
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
                ax1.plot(xs, poly(xs), "r--", alpha=0.8, linewidth=2)

            ax1.set_xlabel(factor, fontsize=12)
            ax1.set_ylabel(target_col, fontsize=12)
            ax1.set_title(
                f'{factor} vs {target_col}\nPearson r = {correlation_value:.3f}, p = {p_value:.4f}',
                fontsize=14, fontweight='bold'
            )
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.hist(x, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            ax2.set_xlabel(factor, fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'Distribution of {factor}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{factor.replace(' ', '_').replace('/', '_')}_correlation.png"
        filepath = os.path.join(self.corr_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        return filepath

    def interval_analysis(self, factor, target_col, n_bins=5, method="quantile"):
        """
        Interval analysis with robust auto-filtering.
        - Never crashes on CI assignment
        - Automatically skips invalid bins
        """

        # ---------- numeric clean ----------
        df = self.data[[factor, target_col]].copy()
        df[factor] = pd.to_numeric(df[factor], errors="coerce")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        df = df.dropna()

        if df.shape[0] < 3:
            self.results[factor] = {
                "interval_analysis": pd.DataFrame(),
                "correlation": np.nan,
                "p_value": np.nan
            }
            return pd.DataFrame()

        # ---------- binning ----------
        try:
            if method == "quantile":
                df["Interval"] = pd.qcut(df[factor], q=n_bins, duplicates="drop")
            else:
                df["Interval"] = pd.cut(df[factor], bins=n_bins)
        except Exception:
            return pd.DataFrame()

        df = df.dropna(subset=["Interval"])
        if df.shape[0] < 3:
            return pd.DataFrame()

        # ---------- aggregation ----------
        if self.target_type == "binary":
            df[target_col] = df[target_col].round().astype(int)

            interval_stats = (
                df.groupby("Interval")
                .agg(
                    Sample_Count=(factor, "count"),
                    Positive_Count=(target_col, "sum"),
                    Positive_Rate=(target_col, "mean"),
                    Factor_Mean=(factor, "mean"),
                    Factor_Min=(factor, "min"),
                    Factor_Max=(factor, "max"),
                )
                .reset_index()
            )
        else:
            interval_stats = (
                df.groupby("Interval")
                .agg(
                    Sample_Count=(factor, "count"),
                    Target_Mean=(target_col, "mean"),
                    Target_Std=(target_col, "std"),
                    Factor_Mean=(factor, "mean"),
                    Factor_Min=(factor, "min"),
                    Factor_Max=(factor, "max"),
                )
                .reset_index()
            )

        # ---------- CI columns (always aligned) ----------
        interval_stats["CI_Lower"] = np.nan
        interval_stats["CI_Upper"] = np.nan

        # ---------- CI calculation ----------
        if self.target_type == "binary":
            from statsmodels.stats.proportion import proportion_confint

            for idx, row in interval_stats.iterrows():
                n = int(row["Sample_Count"])
                p = row["Positive_Rate"]

                if n > 0 and np.isfinite(p):
                    try:
                        ci_l, ci_u = proportion_confint(
                            count=int(round(p * n)),
                            nobs=n,
                            alpha=0.05,
                            method="wilson",
                        )
                        interval_stats.at[idx, "CI_Lower"] = ci_l
                        interval_stats.at[idx, "CI_Upper"] = ci_u
                    except Exception:
                        pass

            # correlation
            if np.nanstd(df[factor]) > 1e-12 and np.nanstd(df[target_col]) > 1e-12:
                r, pval = pointbiserialr(df[factor], df[target_col])
            else:
                r, pval = np.nan, np.nan

        else:
            for idx, row in interval_stats.iterrows():
                n = int(row["Sample_Count"])
                mean = row["Target_Mean"]
                std = row["Target_Std"]

                if n > 1 and np.isfinite(mean) and np.isfinite(std):
                    se = std / np.sqrt(n)
                    interval_stats.at[idx, "CI_Lower"] = mean - 1.96 * se
                    interval_stats.at[idx, "CI_Upper"] = mean + 1.96 * se

            if np.nanstd(df[factor]) > 1e-12 and np.nanstd(df[target_col]) > 1e-12:
                r, pval = pearsonr(df[factor], df[target_col])
            else:
                r, pval = np.nan, np.nan

        self.results[factor] = {
            "interval_analysis": interval_stats,
            "correlation": r,
            "p_value": pval,
        }

        return interval_stats

    def plot_interval_analysis_single(self, factor, target_col, n_bins=5, method='quantile'):
        """Plot interval analysis for a single factor with robust handling for binary/continuous targets."""
        # Run interval analysis
        result = self.interval_analysis(factor, target_col, n_bins=n_bins, method=method)

        # Create figure early for consistent output
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax1, ax2 = axes

        # If empty, create an informative plot and return
        if result is None or not isinstance(result, pd.DataFrame) or result.shape[0] == 0:
            for ax in axes:
                ax.text(0.5, 0.5, "No interval results (insufficient numeric data)", ha="center", va="center")
                ax.axis("off")
            plt.tight_layout()
            filename = f"{factor.replace(' ', '_').replace('/', '_')}_interval.png"
            filepath = os.path.join(self.interval_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            return filepath, result

        # Determine which y-column to plot
        if 'Target_Mean' in result.columns:
            y_col = 'Target_Mean'
            y_label = f"Mean {target_col}"
            title_left = f"Interval Mean of {target_col}"
            y_err = result['Target_Std'].values if 'Target_Std' in result.columns else None
        elif 'Positive_Rate' in result.columns:
            y_col = 'Positive_Rate'
            y_label = "Positive Rate"
            title_left = "Interval Positive Rate"
            y_err = None
        else:
            cand = [c for c in result.columns
                    if c not in ['Interval', 'Sample_Count', 'Sample_Percentage']
                    and pd.api.types.is_numeric_dtype(result[c])]
            y_col = cand[0] if cand else None
            y_label = y_col or "Value"
            title_left = "Interval Summary"
            y_err = None

        # Prepare x labels
        interval_labels = [str(x) for x in result['Interval']]
        x_pos = np.arange(len(interval_labels))

        # Left: bar plot
        if y_col is None:
            ax1.text(0.5, 0.5, "No plottable metric column found", ha="center", va="center")
            ax1.axis("off")
        else:
            bars = ax1.bar(x_pos, result[y_col].values, alpha=0.8)
            if y_err is not None:
                ax1.errorbar(x_pos, result[y_col].values, yerr=y_err, fmt='none', capsize=4)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(interval_labels, rotation=45, ha='right')
            ax1.set_xlabel(f"{factor} intervals", fontsize=12)
            ax1.set_ylabel(y_label, fontsize=12)
            ax1.set_title(title_left, fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                h = bar.get_height()
                if np.isfinite(h):
                    ax1.text(bar.get_x() + bar.get_width()/2, h, f"{h:.3g}", ha='center', va='bottom', fontsize=9)

        # Right: sample distribution per interval
        if 'Sample_Count' in result.columns:
            ax2.bar(x_pos, result['Sample_Count'].values, alpha=0.8)
            ax2.set_ylabel("Sample Count", fontsize=12)
        else:
            ax2.bar(x_pos, np.ones_like(x_pos), alpha=0.2)
            ax2.set_ylabel("Count", fontsize=12)

        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(interval_labels, rotation=45, ha='right')
        ax2.set_xlabel(f"{factor} intervals", fontsize=12)
        ax2.set_title("Samples per Interval", fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f"{factor.replace(' ', '_').replace('/', '_')}_interval.png"
        filepath = os.path.join(self.interval_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath, result

    def prepare_data(self, factor_cols, target_col, test_size=0.2, random_state=42):
        """Prepare data for machine learning (robust to object/string values)."""
        df = self.data[factor_cols + [target_col]].copy()

        for c in factor_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

        data_clean = df.dropna()

        # 调试信息：打印清理后的数据状态
        print(f"\n[Debug] After numeric conversion and dropna:")
        print(f"  Original rows: {len(self.data)}")
        print(f"  After dropna: {len(data_clean)}")
        print(f"  Factor columns: {len(factor_cols)}")

        if len(data_clean) == 0:
            print("\n[ERROR] No valid samples after cleaning!")
            print("  This means all values are either NaN or non-numeric.")
            print("  Please check your Excel file data format.")
            return {
                'X_train': np.array([]).reshape(0, len(factor_cols)),
                'X_test': np.array([]).reshape(0, len(factor_cols)),
                'y_train': np.array([]),
                'y_test': np.array([]),
                'scaler': StandardScaler(),
                'feature_names': factor_cols,
                'target_type': self.target_type
            }

        if self.target_type == "binary":
            data_clean[target_col] = data_clean[target_col].round().astype(int)
            class_counts = data_clean[target_col].value_counts()
            print(f"  Class distribution: {dict(class_counts)}")

            if len(class_counts) < 2:
                print("\n[ERROR] Binary target has only one class!")
                print(f"  Found classes: {class_counts.index.tolist()}")
                print("  Binary classification requires both 0 and 1 classes.")
                print("  Please check if target column contains only 0s or only 1s.")

            if 0 in class_counts.index and 1 in class_counts.index:
                print(f"  Class ratio (1:0): {class_counts.get(1, 0)/class_counts.get(0, 1):.2f}")
            else:
                print(f"  WARNING: Class imbalance or single class detected!")

        X = data_clean[factor_cols].values.astype(float)
        y = data_clean[target_col].values.astype(float)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if self.target_type == "binary":
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=random_state
            )

        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'scaler': scaler, 'feature_names': factor_cols,
            'target_type': self.target_type
        }

    def train_models(self, factor_cols, target_col):
        """Train machine learning models based on target type"""
        data_dict = self.prepare_data(factor_cols, target_col)

        if data_dict['X_train'].shape[0] == 0 or data_dict['X_test'].shape[0] == 0:
            print("\n" + "="*70)
            print("ERROR: No valid samples for model training!")
            print("="*70)
            print("\nPossible causes:")
            print("  1. Too many missing values in factor columns")
            print("  2. Non-numeric values in factor columns")
            print("  3. Target column has too many missing values")
            print("  4. All samples belong to only one class (binary target)")

            print(f"\nFactor columns requested: {len(factor_cols)}")
            print(f"Total rows in data: {len(self.data)}")

            for col in factor_cols[:5]:
                missing_count = self.data[col].isnull().sum()
                non_numeric = pd.to_numeric(self.data[col], errors="coerce").isnull().sum()
                print(f"  {col}: {missing_count} NaN, {non_numeric - missing_count} non-numeric")

            print("\nSkipping model training due to insufficient valid data.")
            return pd.DataFrame()

        print(f"\nValid samples: {data_dict['X_train'].shape[0]} train, {data_dict['X_test'].shape[0]} test")

        if self.target_type == "binary":
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
                'Ridge Classifier': RidgeClassifier(random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
            }
        else:
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }

        results = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(data_dict['X_train'], data_dict['y_train'])

            y_pred = model.predict(data_dict['X_test'])

            if self.target_type == "binary":
                y_pred_proba = model.predict_proba(data_dict['X_test'])[:, 1] if hasattr(model, 'predict_proba') else None

                accuracy = accuracy_score(data_dict['y_test'], y_pred)
                precision = precision_score(data_dict['y_test'], y_pred)
                recall = recall_score(data_dict['y_test'], y_pred)
                f1 = f1_score(data_dict['y_test'], y_pred)
                roc_auc = roc_auc_score(data_dict['y_test'], y_pred_proba) if y_pred_proba is not None else np.nan

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, data_dict['X_train'], data_dict['y_train'],
                                           cv=cv, scoring='roc_auc')

                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': data_dict['y_test']
                }
            else:
                mse = mean_squared_error(data_dict['y_test'], y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(data_dict['y_test'], y_pred)
                r2 = r2_score(data_dict['y_test'], y_pred)

                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(model, data_dict['X_train'], data_dict['y_train'],
                                           cv=cv, scoring='r2')

                results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': data_dict['y_test']
                }

            self.models[name] = model
            self.plot_model_results_single(name, model, results[name])

        results_list = []
        for name, result in results.items():
            if self.target_type == "binary":
                results_list.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1_Score': result['f1_score'],
                    'ROC_AUC': result['roc_auc'],
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std']
                })
            else:
                results_list.append({
                    'Model': name,
                    'R2': result['r2'],
                    'MSE': result['mse'],
                    'RMSE': result['rmse'],
                    'MAE': result['mae'],
                    'CV_Mean': result['cv_mean'],
                    'CV_Std': result['cv_std']
                })

        results_df = pd.DataFrame(results_list)
        self.model_results_df = results_df

        return results_df

    def plot_model_results_single(self, model_name, model, results):
        """Plot results for a single model"""
        if self.target_type == "binary":
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            ax1 = axes[0, 0]
            cm = confusion_matrix(results['y_test'], results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])
            ax1.set_xlabel('Predicted', fontsize=12)
            ax1.set_ylabel('Actual', fontsize=12)
            ax1.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')

            ax2 = axes[0, 1]
            if 'y_pred_proba' in results and results['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
                roc_auc = auc(fpr, tpr)

                ax2.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.3f})')
                ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax2.set_xlim([0.0, 1.0])
                ax2.set_ylim([0.0, 1.05])
                ax2.set_xlabel('False Positive Rate', fontsize=12)
                ax2.set_ylabel('True Positive Rate', fontsize=12)
                ax2.set_title(f'{model_name} - ROC Curve', fontsize=14, fontweight='bold')
                ax2.legend(loc="lower right")
                ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            if 'y_pred_proba' in results and results['y_pred_proba'] is not None:
                precision_vals, recall_vals, _ = precision_recall_curve(results['y_test'], results['y_pred_proba'])
                ax3.plot(recall_vals, precision_vals, color='green', lw=2)
                ax3.set_xlabel('Recall', fontsize=12)
                ax3.set_ylabel('Precision', fontsize=12)
                ax3.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14, fontweight='bold')
                ax3.set_xlim([0.0, 1.0])
                ax3.set_ylim([0.0, 1.05])
                ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                indices = np.argsort(feature_importance)

                ax4.barh(range(len(indices)), feature_importance[indices], color='purple', alpha=0.7)
                ax4.set_yticks(range(len(indices)))
                ax4.set_yticklabels([self.data.columns[i] for i in indices])
                ax4.set_xlabel('Importance', fontsize=12)
                ax4.set_title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)

        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            ax1 = axes[0, 0]
            ax1.scatter(results['y_test'], results['y_pred'], alpha=0.6, color='steelblue')

            min_val = min(results['y_test'].min(), results['y_pred'].min())
            max_val = max(results['y_test'].max(), results['y_pred'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

            ax1.set_xlabel('Actual Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title(f'{model_name} - Actual vs Predicted\nR2 = {results["r2"]:.3f}',
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            ax2 = axes[0, 1]
            residuals = results['y_test'] - results['y_pred']
            ax2.scatter(results['y_pred'], residuals, alpha=0.6, color='green')
            ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Predicted Values', fontsize=12)
            ax2.set_ylabel('Residuals', fontsize=12)
            ax2.set_title(f'{model_name} - Residual Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            ax3 = axes[1, 0]
            ax3.hist(residuals, bins=30, color='orange', edgecolor='black', alpha=0.7)
            ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax3.set_xlabel('Residuals', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title(f'{model_name} - Residual Distribution', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

            ax4 = axes[1, 1]
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                indices = np.argsort(feature_importance)

                ax4.barh(range(len(indices)), feature_importance[indices], color='purple', alpha=0.7)
                ax4.set_yticks(range(len(indices)))
                ax4.set_yticklabels([self.data.columns[i] for i in indices])
                ax4.set_xlabel('Importance', fontsize=12)
                ax4.set_title(f'{model_name} - Feature Importance', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = f"{model_name.replace(' ', '_').lower()}_results.png"
        filepath = os.path.join(self.model_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    # ==================== Feature Importance Analysis ====================

    def feature_importance_analysis(self, factor_cols, target_col):
        """Feature importance analysis based on target type"""
        data_dict = self.prepare_data(factor_cols, target_col)

        if data_dict['X_train'].shape[0] == 0:
            print("\n[ERROR] No valid data for feature importance analysis.")
            return pd.DataFrame()

        print(f"\nAnalyzing feature importance for {self.target_type} target...")

        if self.target_type == "binary":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(data_dict['X_train'], data_dict['y_train'])

            rf_importance = pd.DataFrame({
                'Feature': factor_cols,
                'Importance_RF': rf.feature_importances_
            }).sort_values('Importance_RF', ascending=False)

            gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
            gb.fit(data_dict['X_train'], data_dict['y_train'])

            gb_importance = pd.DataFrame({
                'Feature': factor_cols,
                'Importance_GB': gb.feature_importances_
            }).sort_values('Importance_GB', ascending=False)

            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(data_dict['X_train'], data_dict['y_train'])

            f_scores = pd.DataFrame({
                'Feature': factor_cols,
                'F_Score': selector_f.scores_
            }).sort_values('F_Score', ascending=False)

            selector_mi = SelectKBest(score_func=mutual_info_classif, k='all')
            selector_mi.fit(data_dict['X_train'], data_dict['y_train'])

            mi_scores = pd.DataFrame({
                'Feature': factor_cols,
                'MI_Score': selector_mi.scores_
            }).sort_values('MI_Score', ascending=False)
        else:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(data_dict['X_train'], data_dict['y_train'])

            rf_importance = pd.DataFrame({
                'Feature': factor_cols,
                'Importance_RF': rf.feature_importances_
            }).sort_values('Importance_RF', ascending=False)

            gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
            gb.fit(data_dict['X_train'], data_dict['y_train'])

            gb_importance = pd.DataFrame({
                'Feature': factor_cols,
                'Importance_GB': gb.feature_importances_
            }).sort_values('Importance_GB', ascending=False)

            selector_f = SelectKBest(score_func=f_regression, k='all')
            selector_f.fit(data_dict['X_train'], data_dict['y_train'])

            f_scores = pd.DataFrame({
                'Feature': factor_cols,
                'F_Score': selector_f.scores_
            }).sort_values('F_Score', ascending=False)

            selector_mi = SelectKBest(score_func=mutual_info_regression, k='all')
            selector_mi.fit(data_dict['X_train'], data_dict['y_train'])

            mi_scores = pd.DataFrame({
                'Feature': factor_cols,
                'MI_Score': selector_mi.scores_
            }).sort_values('MI_Score', ascending=False)

        importance_df = rf_importance.merge(gb_importance, on='Feature')
        importance_df = importance_df.merge(f_scores, on='Feature')
        importance_df = importance_df.merge(mi_scores, on='Feature')

        for col in ['Importance_RF', 'Importance_GB', 'F_Score', 'MI_Score']:
            importance_df[f'{col}_Norm'] = (importance_df[col] - importance_df[col].min()) / \
                                          (importance_df[col].max() - importance_df[col].min())

        importance_df['Composite_Score'] = importance_df[[
            'Importance_RF_Norm', 'Importance_GB_Norm',
            'F_Score_Norm', 'MI_Score_Norm'
        ]].mean(axis=1)

        importance_df = importance_df.sort_values('Composite_Score', ascending=False)

        self.feature_importance = importance_df

        self.plot_feature_importance_single(importance_df)

        return importance_df

    def plot_feature_importance_single(self, importance_df, top_n_summary=50):
        """Plot feature importance for each feature separately and summary plot."""
        features = importance_df['Feature'].tolist()

        for idx, feature in enumerate(features):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            row = importance_df[importance_df['Feature'] == feature].iloc[0]

            ax1 = axes[0]
            scores = {
                'RF Importance': row['Importance_RF'],
                'GB Importance': row['Importance_GB'],
                'F-Score': row['F_Score'],
                'MI Score': row['MI_Score']
            }
            colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
            bars = ax1.bar(scores.keys(), scores.values(), color=colors)
            ax1.set_ylabel('Score', fontsize=12)
            ax1.set_title(f'{feature} - Individual Importance Scores', fontsize=14, fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            for bar, value in zip(bars, scores.values()):
                ax1.text(bar.get_x() + bar.get_width()/2., value + 0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=10)

            ax2 = axes[1]
            ax2.bar(['Composite Score'], [row['Composite_Score']], color='purple', alpha=0.7)
            ax2.set_ylabel('Composite Score (Normalized)', fontsize=12)
            ax2.set_title(f'{feature} - Composite Score: {row["Composite_Score"]:.4f}', fontsize=14, fontweight='bold')
            ax2.set_ylim([0, 1])
            ax2.grid(True, alpha=0.3)
            ax2.text(0, row['Composite_Score'] + 0.02, f'{row["Composite_Score"]:.4f}', ha='center', va='bottom', fontsize=12)

            plt.tight_layout()

            filename = f"{feature.replace(' ', '_').replace('/', '_')}_importance.png"
            filepath = os.path.join(self.feature_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

        importance_summary = importance_df.head(top_n_summary)
        fig_height = min(0.3 * len(importance_summary), 20)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y_pos = np.arange(len(importance_summary))
        ax.barh(y_pos, importance_summary['Composite_Score'], color='steelblue', alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_summary['Feature'])
        ax.set_xlabel('Composite Importance Score', fontsize=12)
        ax.set_title('Feature Importance Ranking (Top {})'.format(top_n_summary), fontsize=16, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)

        for i, v in enumerate(importance_summary['Composite_Score']):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

        plt.tight_layout()
        filename = "feature_importance_summary.png"
        filepath = os.path.join(self.plot_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filepath

    # ==================== Advanced Analysis Methods ====================

    def create_all_individual_plots(self, factor_cols, target_col):
        """Create all individual plots for each factor"""
        print("\n" + "="*70)
        print("CREATING INDIVIDUAL ANALYSIS PLOTS")
        print("="*70)

        corr_matrix, target_corr, p_values = self.calculate_correlations(factor_cols, target_col)

        plot_files = []

        for factor in factor_cols:
            print(f"Creating plots for {factor}...")

            corr_value = target_corr.get(factor, np.nan)
            p_val = p_values.get(factor, np.nan)
            corr_plot = self.plot_correlation_single(factor, target_col, corr_value, p_val)

            interval_plot, interval_results = self.plot_interval_analysis_single(factor, target_col, n_bins=5)

            plot_files.append({
                'factor': factor,
                'correlation_plot': corr_plot,
                'interval_plot': interval_plot,
                'correlation': corr_value,
                'p_value': p_val
            })

            print(f"  - Created: {os.path.basename(corr_plot)}")
            print(f"  - Created: {os.path.basename(interval_plot)}")

        return plot_files

    def create_summary_plots(self, factor_cols, target_col):
        """Create summary plots (robust to non-numeric values)."""
        print("\nCreating summary plots...")

        corr_matrix, target_corr, p_values = self.calculate_correlations(factor_cols, target_col)

        fig, ax = plt.subplots(figsize=(10, 6))
        factors = list(target_corr.index)
        correlations = target_corr.values

        sorted_indices = np.argsort(np.abs(correlations))[::-1]
        factors_sorted = [factors[i] for i in sorted_indices]
        correlations_sorted = correlations[sorted_indices]

        colors = ['red' if x > 0 else 'blue' for x in correlations_sorted]
        bars = ax.barh(range(len(factors_sorted)), correlations_sorted, color=colors, alpha=0.7)
        ax.set_yticks(range(len(factors_sorted)))
        ax.set_yticklabels(factors_sorted)

        ax.set_xlabel('Point-biserial Correlation Coefficient' if self.target_type == "binary" else 'Pearson Correlation Coefficient', fontsize=12)
        ax.set_title(f'Correlation with {target_col}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        for bar, corr in zip(bars, correlations_sorted):
            ax.text(bar.get_width() + (0.01 if corr > 0 else -0.05),
                    bar.get_y() + bar.get_height()/2,
                    f'{corr:.3f}',
                    va='center', ha='left' if corr > 0 else 'right', fontsize=10)

        plt.tight_layout()
        summary_plot1 = os.path.join(self.plot_dir, "correlation_summary.png")
        plt.savefig(summary_plot1, dpi=300, bbox_inches='tight')
        plt.close()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax1, ax2 = axes

        if self.target_type == "binary":
            target_num = pd.to_numeric(self.data[target_col], errors="coerce").dropna().round().astype(int)
            n_positive = int((target_num == 1).sum())
            n_negative = int((target_num == 0).sum())
            total = int(len(target_num))

            labels = ['Class 0', 'Class 1']
            sizes = [n_negative, n_positive]
            colors = ['lightblue', 'lightcoral']
            explode = (0.05, 0.05)

            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.axis('equal')
            ax1.set_title(f'Class Distribution\nTotal samples: {total}', fontsize=14, fontweight='bold')

            factor = factor_cols[0] if factor_cols else None
            if factor:
                df = self._numeric_df([factor, target_col]).dropna()
                if len(df):
                    df[target_col] = df[target_col].round().astype(int)
                    box_data = [df[df[target_col] == 0][factor].values,
                                df[df[target_col] == 1][factor].values]
                    box = ax2.boxplot(box_data, labels=['Class 0', 'Class 1'], patch_artist=True)
                    colors2 = ['lightblue', 'lightcoral']
                    for patch, color in zip(box['boxes'], colors2):
                        patch.set_facecolor(color)
                    ax2.set_ylabel(factor, fontsize=12)
                    ax2.set_title(f'{factor} by Target Class', fontsize=14, fontweight='bold')
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, "No numeric data", ha="center", va="center")
                    ax2.axis("off")
        else:
            target_num = pd.to_numeric(self.data[target_col], errors="coerce").dropna()
            vals = target_num.values.astype(float)

            ax1.hist(vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.axvline(float(target_num.mean()), color='red', linestyle='--', linewidth=2, label=f'Mean: {target_num.mean():.2f}')
            ax1.axvline(float(target_num.median()), color='green', linestyle='--', linewidth=2, label=f'Median: {target_num.median():.2f}')
            ax1.set_xlabel(target_col, fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.set_title(f'{target_col} Distribution', fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.boxplot(vals, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax2.set_xlabel(target_col, fontsize=12)
            ax2.set_title(f'{target_col} Box Plot', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        summary_plot2 = os.path.join(self.plot_dir, "target_distribution.png")
        plt.savefig(summary_plot2, dpi=300, bbox_inches='tight')
        plt.close()

        return summary_plot1, summary_plot2

    def _clean_sheet_name(self, name):
        """清理工作表名称，移除Excel不允许的字符"""
        illegal_chars = r'[/\\*?:\[\]]'
        import re
        clean_name = re.sub(illegal_chars, '_', name)

        clean_name = clean_name.strip("'")

        if len(clean_name) > 31:
            clean_name = clean_name[:31]

        if not clean_name:
            clean_name = "Sheet"

        return clean_name

    def export_results(self, factor_cols, target_col):
        """Export all analysis results to Excel"""
        output_file = os.path.join(self.output_dir, f"{target_col}_analysis_results.xlsx")

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            overview_df = pd.DataFrame({
                'Description': [
                    'Analysis Date',
                    'Total Samples',
                    'Target Variable',
                    'Target Type',
                    'Number of Factors Analyzed'
                ],
                'Value': [
                    pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(self.data),
                    target_col,
                    self.target_type,
                    len(factor_cols)
                ]
            })
            overview_df.to_excel(writer, sheet_name='Data_Overview', index=False)

            desc_stats = self.basic_descriptive_stats(factor_cols, target_col)
            desc_stats.to_excel(writer, sheet_name='Descriptive_Statistics')

            corr_matrix, target_corr, p_values = self.calculate_correlations(factor_cols, target_col)
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')

            target_corr_df = pd.DataFrame({
                'Factor': target_corr.index,
                'Correlation': target_corr.values,
                'P_Value': [p_values.get(col, np.nan) for col in target_corr.index],
                'Significance': ['***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                                for p in [p_values.get(col, np.nan) for col in target_corr.index]]
            })
            target_corr_df.to_excel(writer, sheet_name='Target_Correlations', index=False)

            for factor in factor_cols:
                if factor in self.results:
                    result = self.results[factor]['interval_analysis']
                    result.insert(0, 'Factor', factor)
                    result.insert(1, 'Correlation', self.results[factor]['correlation'])
                    result.insert(2, 'P_Value', self.results[factor]['p_value'])

                    sheet_name = self._clean_sheet_name(f'Interval_{factor}')

                    base_name = sheet_name
                    counter = 1
                    while sheet_name in writer.book.sheetnames:
                        sheet_name = f"{base_name}_{counter}"
                        counter += 1
                        if counter > 100:
                            sheet_name = f"Interval_{counter}"
                            break

                    result.to_excel(writer, sheet_name=sheet_name, index=False)

            if self.models:
                if self.model_results_df is not None:
                    model_results_df = self.model_results_df
                else:
                    model_results_df = self.train_models(factor_cols, target_col)

                metrics_df = model_results_df.copy()
                metrics_df.to_excel(writer, sheet_name='Model_Comparison', index=False)

            if self.feature_importance is not None:
                importance_df = self.feature_importance_analysis(factor_cols, target_col)
                importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

            key_findings = []

            pos_corr_factors = target_corr_df[target_corr_df['Correlation'] > 0]
            if len(pos_corr_factors) > 0:
                strongest_pos = pos_corr_factors.loc[pos_corr_factors['Correlation'].idxmax()]
                key_findings.append({
                    'Finding': 'Strongest positive correlation',
                    'Factor': strongest_pos['Factor'],
                    'Value': f"r = {strongest_pos['Correlation']:.3f} (p = {strongest_pos['P_Value']:.4e})",
                    'Interpretation': 'Higher values associated with increased target values'
                })

            neg_corr_factors = target_corr_df[target_corr_df['Correlation'] < 0]
            if len(neg_corr_factors) > 0:
                strongest_neg = neg_corr_factors.loc[neg_corr_factors['Correlation'].idxmin()]
                key_findings.append({
                    'Finding': 'Strongest negative correlation',
                    'Factor': strongest_neg['Factor'],
                    'Value': f"r = {strongest_neg['Correlation']:.3f} (p = {strongest_neg['P_Value']:.4e})",
                    'Interpretation': 'Higher values associated with decreased target values'
                })

            if self.feature_importance is not None:
                top_feature = self.feature_importance.iloc[0]
                key_findings.append({
                    'Finding': 'Most important feature (ML models)',
                    'Factor': top_feature['Feature'],
                    'Value': f"Composite score: {top_feature['Composite_Score']:.3f}",
                    'Interpretation': 'Most predictive factor for target based on ML analysis'
                })

            if self.models and self.model_results_df is not None:
                if self.target_type == "binary":
                    best_model_idx = self.model_results_df['ROC_AUC'].idxmax()
                else:
                    best_model_idx = self.model_results_df['R2'].idxmax()

                best_model = self.model_results_df.loc[best_model_idx]
                key_findings.append({
                    'Finding': 'Best performing model',
                    'Factor': best_model['Model'],
                    'Value': f"Score: {best_model['ROC_AUC' if self.target_type == 'binary' else 'R2']:.3f}",
                    'Interpretation': 'Most accurate model for predicting target'
                })

            key_findings_df = pd.DataFrame(key_findings)
            key_findings_df.to_excel(writer, sheet_name='Key_Findings', index=False)

            if len(self.data) > 1000:
                self.data.head(1000).to_excel(writer, sheet_name='Original_Data', index=False)
            else:
                self.data.to_excel(writer, sheet_name='Original_Data', index=False)

        print(f"\nAll analysis results exported to: {output_file}")
        return output_file

    def generate_report(self, factor_cols, target_col):
        """Generate comprehensive analysis report"""
        report_lines = []

        report_lines.append("=" * 70)
        report_lines.append("PEPTIDE PROPERTY ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Analysis Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Samples: {len(self.data)}")
        report_lines.append(f"Analyzed Factors: {', '.join(factor_cols)}")
        report_lines.append(f"Target Variable: {target_col} (Type: {self.target_type})")
        report_lines.append("")

        target_data = self.data[target_col].dropna()

        report_lines.append("1. TARGET DISTRIBUTION")
        report_lines.append("-" * 40)

        total = len(target_data)

        if self.target_type == "binary":
            n_positive = (target_data == 1).sum()
            n_negative = (target_data == 0).sum()

            report_lines.append(f"Class 1 (Positive): {n_positive} ({n_positive/total*100:.1f}%)")
            report_lines.append(f"Class 0 (Negative): {n_negative} ({n_negative/total*100:.1f}%)")
        else:
            report_lines.append(f"Mean: {target_data.mean():.2f}")
            report_lines.append(f"Std: {target_data.std():.2f}")
            report_lines.append(f"Min: {target_data.min():.2f}")
            report_lines.append(f"25%: {target_data.quantile(0.25):.2f}")
            report_lines.append(f"Median: {target_data.median():.2f}")
            report_lines.append(f"75%: {target_data.quantile(0.75):.2f}")
            report_lines.append(f"Max: {target_data.max():.2f}")

        report_lines.append(f"Total (non-missing): {total}")
        report_lines.append(f"Missing values: {self.data[target_col].isnull().sum()}")
        report_lines.append("")

        report_lines.append("2. CORRELATION ANALYSIS")
        report_lines.append("-" * 40)
        _, target_corr, p_values = self.calculate_correlations(factor_cols, target_col)

        for factor in factor_cols:
            if factor in target_corr.index:
                corr = target_corr[factor]
                p_val = p_values.get(factor, np.nan)

                sig_star = ""
                if p_val < 0.001:
                    sig_star = "***"
                elif p_val < 0.01:
                    sig_star = "**"
                elif p_val < 0.05:
                    sig_star = "*"

                direction = "positive" if corr > 0 else "negative"
                interpretation = ""
                if abs(corr) > 0.5:
                    interpretation = "(strong correlation)"
                elif abs(corr) > 0.3:
                    interpretation = "(moderate correlation)"
                elif abs(corr) > 0.1:
                    interpretation = "(weak correlation)"

                report_lines.append(f"{factor}:")
                report_lines.append(f"  r = {corr:.4f}{sig_star} (p = {p_val:.4e})")
                report_lines.append(f"  Direction: {direction} {interpretation}")
        report_lines.append("")

        report_lines.append("3. KEY FINDINGS")
        report_lines.append("-" * 40)

        if len(target_corr) > 0:
            strongest_pos = target_corr[target_corr > 0].max() if any(target_corr > 0) else None
            strongest_neg = target_corr[target_corr < 0].min() if any(target_corr < 0) else None

            if strongest_pos is not None:
                pos_factor = target_corr[target_corr == strongest_pos].index[0]
                report_lines.append(f"* Strongest positive correlation: {pos_factor}")
                report_lines.append(f"  (r = {strongest_pos:.3f}, p = {p_values.get(pos_factor, np.nan):.4e})")

            if strongest_neg is not None:
                neg_factor = target_corr[target_corr == strongest_neg].index[0]
                report_lines.append(f"* Strongest negative correlation: {neg_factor}")
                report_lines.append(f"  (r = {strongest_neg:.3f}, p = {p_values.get(neg_factor, np.nan):.4e})")

        report_lines.append("")

        report_lines.append("4. RECOMMENDATIONS")
        report_lines.append("-" * 40)
        if self.target_type == "binary":
            report_lines.append("* Factors with strong positive correlation may promote positive class")
            report_lines.append("* Factors with strong negative correlation may inhibit positive class")
        else:
            report_lines.append("* Factors with strong positive correlation increase target values")
            report_lines.append("* Factors with strong negative correlation decrease target values")
        report_lines.append("* Consider adjusting these factors in peptide design")
        report_lines.append("* Validate findings with additional experimental data")

        report_lines.append("")
        report_lines.append("=" * 70)
        report_lines.append("ANALYSIS COMPLETE")
        report_lines.append("=" * 70)

        report_text = "\n".join(report_lines)

        report_file = os.path.join(self.output_dir, f"{target_col}_analysis_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)

        md_file = os.path.join(self.output_dir, f"{target_col}_analysis_report.md")
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(report_text.replace("=", "#").replace("-", "---"))

        return report_text


# ==================== Main Program ====================

def get_file_path():
    """Get file path from user input"""
    print("\n" + "="*70)
    print("PEPTIDE PROPERTY ANALYSIS SYSTEM")
    print("="*70)

    excel_files = [f for f in os.listdir('.') if f.lower().endswith(('.xlsx', '.xls'))]

    if excel_files:
        print("\nExcel files found in current directory:")
        for i, f in enumerate(excel_files, 1):
            print(f"  {i}. {f}")
        print("  - Enter '0' to specify a different file")

    file_choice = input("\nEnter file number or full path to Excel file: ").strip()

    if file_choice.isdigit():
        choice_num = int(file_choice)
        if 1 <= choice_num <= len(excel_files):
            file_path = excel_files[choice_num - 1]
        else:
            file_path = input("Enter full path to Excel file: ").strip()
    else:
        file_path = file_choice

    if not os.path.exists(file_path):
        print(f"\nERROR: File not found: {file_path}")
        return get_file_path()

    return file_path


# ==================== MODIFIED SECTION: MIC two-class classification (Low/High) with Mann-Whitney U test ====================
def add_mic_group(self, target_col="Exp_Log2_MIC", low_th=3.0, group_col="MIC_Group"):
    """
    Create MIC activity groups from continuous Log2(MIC).
    MODIFIED: Two-class classification: Low (<= low_th) and High (> low_th).
    Default threshold: Low <= 3, High > 3.
    """
    if target_col not in self.data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data.")

    mic = pd.to_numeric(self.data[target_col], errors="coerce")

    # Two bins: [-inf, low_th], (low_th, inf]
    self.data[group_col] = pd.cut(
        mic,
        bins=[-np.inf, low_th, np.inf],
        labels=["Low", "High"],
        right=True,
        include_lowest=True
    )
    return group_col


def _mic_group_summary_one(self, feature, group_col="MIC_Group"):
    """Return per-group descriptive statistics for one feature."""
    df = self.data[[feature, group_col]].copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna()
    if df.empty:
        return pd.DataFrame()

    cat_order = ["Low", "High"]  # Two-class order
    df[group_col] = pd.Categorical(df[group_col], categories=cat_order, ordered=True)

    summary = (
        df.groupby(group_col, observed=True)[feature]
        .agg(
            N="count",
            Mean="mean",
            Median="median",
            Std="std",
            Q10=lambda x: x.quantile(0.10),
            Q25=lambda x: x.quantile(0.25),
            Q75=lambda x: x.quantile(0.75),
            Q90=lambda x: x.quantile(0.90),
            Min="min",
            Max="max"
        )
        .reset_index()
    )
    summary.insert(0, "Feature", feature)
    return summary


def _mic_group_tests_one(self, feature, group_col="MIC_Group"):
    """Mann-Whitney U test between Low and High MIC groups for one feature."""
    df = self.data[[feature, group_col]].copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna()
    if df.empty:
        return {"U_stat": np.nan, "p_value": np.nan, "N_total": 0, "N_low": 0, "N_high": 0}

    low_vals = df.loc[df[group_col] == "Low", feature].values.astype(float)
    high_vals = df.loc[df[group_col] == "High", feature].values.astype(float)

    n_low = len(low_vals)
    n_high = len(high_vals)
    n_total = n_low + n_high

    if n_low == 0 or n_high == 0 or n_total < 3:
        return {"U_stat": np.nan, "p_value": np.nan, "N_total": n_total, "N_low": n_low, "N_high": n_high}

    if np.nanstd(np.concatenate([low_vals, high_vals])) < 1e-12:
        return {"U_stat": np.nan, "p_value": np.nan, "N_total": n_total, "N_low": n_low, "N_high": n_high}

    try:
        # Use Mann-Whitney U test (two-sided)
        u_stat, p_val = mannwhitneyu(low_vals, high_vals, alternative='two-sided')
        return {"U_stat": float(u_stat), "p_value": float(p_val), "N_total": n_total, "N_low": n_low, "N_high": n_high}
    except Exception:
        return {"U_stat": np.nan, "p_value": np.nan, "N_total": n_total, "N_low": n_low, "N_high": n_high}


def plot_feature_by_mic_group(self, feature, target_col="Exp_Log2_MIC", group_col="MIC_Group",
                              low_th=3.0, outdir=None):
    """Boxplot (with jitter) of one feature across Low/High MIC groups, with Mann-Whitney U p-value annotation.
    Two-class classification: Low <= low_th, High > low_th.
    Box and jitter colors: Low -> royalblue, High -> indianred."""
    if outdir is None:
        outdir = getattr(self, "mic_group_dir", self.plot_dir)
    os.makedirs(outdir, exist_ok=True)

    if group_col not in self.data.columns:
        self.add_mic_group(target_col=target_col, low_th=low_th, group_col=group_col)

    df = self.data[[feature, group_col]].copy()
    df[feature] = pd.to_numeric(df[feature], errors="coerce")
    df = df.dropna()
    if df.empty:
        return None

    order = ["Low", "High"]
    df[group_col] = pd.Categorical(df[group_col], categories=order, ordered=True)

    test = self._mic_group_tests_one(feature, group_col=group_col)
    pval = test.get("p_value", np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))

    data = [df.loc[df[group_col] == g, feature].values.astype(float) for g in order]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    # 箱体颜色
    box_colors = ['royalblue', 'indianred']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    # 抖动散点，每组使用对应颜色（可适当调低透明度）
    for i, g in enumerate(order, start=1):
        vals = df.loc[df[group_col] == g, feature].values.astype(float)
        if len(vals) == 0:
            continue
        xs = (np.random.rand(len(vals)) - 0.5) * 0.18 + i
        ax.scatter(xs, vals, alpha=0.6, s=18, color=box_colors[i-1])  # 使用与箱体相同的颜色

    ax.set_xlabel(f"{target_col} group (Low ≤ {low_th}, High > {low_th})")
    ax.set_ylabel(feature)
    if np.isfinite(pval):
        ax.set_title(f"{feature} across MIC groups\nMann-Whitney U p = {pval:.3g}")
    else:
        ax.set_title(f"{feature} across MIC groups")

    ax.grid(True, alpha=0.25, axis="y")
    plt.tight_layout()

    fname = f"{feature.replace(' ', '_').replace('/', '_')}_by_MIC_group.png"
    fpath = os.path.join(outdir, fname)
    plt.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close()
    return fpath


def run_mic_group_analysis(self, factor_cols, target_col="Exp_Log2_MIC",
                           low_th=3.0, group_col="MIC_Group",
                           top_k=30):
    """Run MIC-group stratified analysis for all factors (two-class Low/High)."""
    self.add_delta_hcs_features(verbose=False)
    factor_cols = list(factor_cols)
    for derived_col in ["Delta_HCS4", "Delta_HCS3"]:
        if derived_col in self.data.columns and derived_col not in factor_cols:
            factor_cols.append(derived_col)

    self.mic_group_dir = os.path.join(self.output_dir, "MIC_Group_Analysis")
    os.makedirs(self.mic_group_dir, exist_ok=True)
    plot_dir = os.path.join(self.mic_group_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Add group column (two-class)
    self.add_mic_group(target_col=target_col, low_th=low_th, group_col=group_col)

    all_summ = []
    all_tests = []
    plot_paths = {}

    for feat in factor_cols:
        if feat not in self.data.columns:
            continue

        summ = self._mic_group_summary_one(feat, group_col=group_col)
        if not summ.empty:
            all_summ.append(summ)

        t = self._mic_group_tests_one(feat, group_col=group_col)
        t.update({"Feature": feat})
        all_tests.append(t)

        p = self.plot_feature_by_mic_group(
            feat, target_col=target_col, group_col=group_col,
            low_th=low_th, outdir=plot_dir
        )
        if p:
            plot_paths[feat] = p

    summary_long = pd.concat(all_summ, ignore_index=True) if all_summ else pd.DataFrame()
    tests_df = pd.DataFrame(all_tests)

    # FDR correction on p-values (Mann-Whitney U)
    tests_df["q_value"] = np.nan
    try:
        from statsmodels.stats.multitest import multipletests
        mask = tests_df["p_value"].notna()
        if mask.sum() > 0:
            _, qvals, _, _ = multipletests(tests_df.loc[mask, "p_value"].values, method="fdr_bh")
            tests_df.loc[mask, "q_value"] = qvals
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    xlsx1 = os.path.join(self.mic_group_dir, "mic_group_summary_long.xlsx")
    csv1  = os.path.join(self.mic_group_dir, "mic_group_summary_long.csv")
    xlsx2 = os.path.join(self.mic_group_dir, "mic_group_tests.xlsx")
    csv2  = os.path.join(self.mic_group_dir, "mic_group_tests.csv")

    if not summary_long.empty:
        summary_long.to_excel(xlsx1, index=False)
        summary_long.to_csv(csv1, index=False)
    tests_df.to_excel(xlsx2, index=False)
    tests_df.to_csv(csv2, index=False)

    top_df = tests_df.sort_values(["q_value", "p_value"], ascending=[True, True]).head(top_k).copy()
    top_xlsx = os.path.join(self.mic_group_dir, "mic_group_top_features.xlsx")
    top_df.to_excel(top_xlsx, index=False)

    report_md = os.path.join(self.mic_group_dir, "mic_group_report.md")
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# MIC-group stratified analysis report (Two-class: Low/High)\n\n")
        f.write(f"Target: **{target_col}**\n\n")
        f.write(f"Groups: Low ≤ {low_th}, High > {low_th}\n\n")
        f.write("Test: Mann-Whitney U (two-sided), FDR corrected\n\n")
        f.write("## Outputs\n")
        f.write(f"- Summary stats (long, including Q10/Q90): {xlsx1}\n")
        f.write(f"- Group tests (Mann-Whitney U + FDR): {xlsx2}\n")
        f.write(f"- Top features: {top_xlsx}\n\n")
        f.write("## Top features by FDR (q_value)\n\n")
        if len(top_df) > 0:
            show = top_df[["Feature", "U_stat", "p_value", "q_value", "N_total", "N_low", "N_high"]]
            f.write("".join(["\t".join(map(str, show.columns))] + ["\t".join(map(str, r)) for r in show.values.tolist()]))
            f.write("\n\n")
        f.write("## Plot paths (per feature)\n\n")
        for feat in list(top_df["Feature"])[:min(15, len(top_df))]:
            if feat in plot_paths:
                f.write(f"- {feat}: {plot_paths[feat]}\n")

    return {
        "mic_group_dir": self.mic_group_dir,
        "summary_long": xlsx1 if os.path.exists(xlsx1) else None,
        "tests": xlsx2,
        "top_features": top_xlsx,
        "report_md": report_md,
        "plot_dir": plot_dir
    }


def analyze_hydrophobic_face_optimization(self, output_excel=None):
    """
    Hydrophobic Face Optimization Analysis - Q10/Q90 Based Dynamic Ranges
    MODIFIED: Now uses two-class MIC grouping (Low ≤ 3, High > 3) for target range.
    """
    hf_features = {
        'AMP_HCS4_HF': {'name': 'HCS4_HF', 'display_name': 'HCS4_HF'},
        'AMP_HCS3_HF': {'name': 'HCS3_HF', 'display_name': 'HCS3_HF'},
        'HMom_HF_mag': {'name': 'HMom_HF_mag', 'display_name': 'HMom_HF_mag'}
    }

    target_col = "Exp_Log2_MIC"
    group_col = "MIC_Group"
    low_th = 3.0  # Threshold for Low/High classification

    if group_col not in self.data.columns:
        self.add_mic_group(target_col=target_col, low_th=low_th, group_col=group_col)

    mic_groups = self.data[group_col].cat.categories.tolist() if hasattr(self.data[group_col], 'cat') else list(self.data[group_col].unique())

    results = []
    hf_stats = {}

    for feature, info in hf_features.items():
        if feature not in self.data.columns:
            results.append({
                'Feature': feature,
                'Display_Name': info['display_name'],
                'Q10_Range_Min': np.nan,
                'Q10_Range_Max': np.nan,
                'Q90_Range_Min': np.nan,
                'Q90_Range_Max': np.nan,
                'Target_Range': 'N/A',
                'Current_Mean': np.nan,
                'Current_Std': np.nan,
                'Current_Min': np.nan,
                'Current_Max': np.nan,
                'Q10_Group_Mean': np.nan,
                'Q90_Group_Mean': np.nan,
                'Status': 'Column not found'
            })
            continue

        col_data = pd.to_numeric(self.data[feature], errors="coerce").dropna()

        if len(col_data) == 0:
            results.append({
                'Feature': feature,
                'Display_Name': info['display_name'],
                'Q10_Range_Min': np.nan,
                'Q10_Range_Max': np.nan,
                'Q90_Range_Min': np.nan,
                'Q90_Range_Max': np.nan,
                'Target_Range': 'N/A',
                'Current_Mean': np.nan,
                'Current_Std': np.nan,
                'Current_Min': np.nan,
                'Current_Max': np.nan,
                'Q10_Group_Mean': np.nan,
                'Q90_Group_Mean': np.nan,
                'Status': 'No valid data'
            })
            continue

        current_min = col_data.min()
        current_max = col_data.max()
        current_mean = col_data.mean()
        current_std = col_data.std()

        low_group = self.data.loc[self.data[group_col] == 'Low', feature]
        high_group = self.data.loc[self.data[group_col] == 'High', feature]

        low_group = pd.to_numeric(low_group, errors="coerce").dropna()
        high_group = pd.to_numeric(high_group, errors="coerce").dropna()

        if len(low_group) > 0:
            low_q10 = low_group.quantile(0.10)
            low_q90 = low_group.quantile(0.90)
            low_mean = low_group.mean()
        else:
            low_q10 = np.nan
            low_q90 = np.nan
            low_mean = np.nan

        if len(high_group) > 0:
            high_q10 = high_group.quantile(0.10)
            high_q90 = high_group.quantile(0.90)
            high_mean = high_group.mean()
        else:
            high_q10 = np.nan
            high_q90 = np.nan
            high_mean = np.nan

        hf_stats[feature] = {
            'low_group_n': len(low_group),
            'high_group_n': len(high_group),
            'low_q10': low_q10,
            'low_q90': low_q90,
            'low_mean': low_mean,
            'high_q10': high_q10,
            'high_q90': high_q90,
            'high_mean': high_mean
        }

        # Target Range: [Low_Group_Mean, High_Group_Mean]
        if pd.notna(low_mean) and pd.notna(high_mean):
            target_min = round(low_mean, 4)
            target_max = round(high_mean, 4)
            target_range_str = f"{target_min} <= {info['display_name']} <= {target_max}"
        else:
            target_min = np.nan
            target_max = np.nan
            target_range_str = "N/A"

        if pd.isna(target_min) or pd.isna(target_max):
            status = "DATA INSUFFICIENT"
        elif current_mean < target_min:
            status = "TOO LOW (Below Target)"
        elif current_mean > target_max:
            status = "TOO HIGH (Above Target)"
        elif (current_min >= target_min * 0.9) and (current_max <= target_max * 1.1):
            status = "OPTIMAL"
        else:
            status = "NEEDS ADJUSTMENT"

        results.append({
            'Feature': feature,
            'Display_Name': info['display_name'],
            'Q10_Range_Min': round(low_q10, 4) if pd.notna(low_q10) else np.nan,
            'Q10_Range_Max': round(low_q90, 4) if pd.notna(low_q90) else np.nan,
            'Q90_Range_Min': round(high_q10, 4) if pd.notna(high_q10) else np.nan,
            'Q90_Range_Max': round(high_q90, 4) if pd.notna(high_q90) else np.nan,
            'Target_Range': target_range_str,
            'Current_Mean': round(current_mean, 4),
            'Current_Std': round(current_std, 4),
            'Current_Min': round(current_min, 4),
            'Current_Max': round(current_max, 4),
            'Q10_Group_Mean': round(low_mean, 4) if pd.notna(low_mean) else np.nan,
            'Q90_Group_Mean': round(high_mean, 4) if pd.notna(high_mean) else np.nan,
            'Status': status
        })

    results_df = pd.DataFrame(results)

    if output_excel:
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            results_df.to_excel(writer, sheet_name='HF_Optimization', index=False)

            stats_data = []
            for feature, stats in hf_stats.items():
                info = hf_features.get(feature, {'display_name': feature})
                stats_data.append({
                    'Feature': feature,
                    'Display_Name': info['display_name'],
                    'Low_Group_N': stats['low_group_n'],
                    'Low_Group_Q10': round(stats['low_q10'], 4) if pd.notna(stats['low_q10']) else np.nan,
                    'Low_Group_Q90': round(stats['low_q90'], 4) if pd.notna(stats['low_q90']) else np.nan,
                    'Low_Group_Mean': round(stats['low_mean'], 4) if pd.notna(stats['low_mean']) else np.nan,
                    'High_Group_N': stats['high_group_n'],
                    'High_Group_Q10': round(stats['high_q10'], 4) if pd.notna(stats['high_q10']) else np.nan,
                    'High_Group_Q90': round(stats['high_q90'], 4) if pd.notna(stats['high_q90']) else np.nan,
                    'High_Group_Mean': round(stats['high_mean'], 4) if pd.notna(stats['high_mean']) else np.nan,
                })

            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='HF_Statistics', index=False)

            report = "="*70 + "\n"
            report += "HYDROPHOBIC FACE OPTIMIZATION REPORT (Q10/Q90 Based Dynamic Ranges)\n"
            report += "="*70 + "\n\n"
            report += "Target Range Calculation Method:\n"
            report += "  - Target Range = [Low_Group_Mean, High_Group_Mean]\n"
            report += f"  - Low Group: MIC ≤ {low_th} (Low activity peptides)\n"
            report += f"  - High Group: MIC > {low_th} (High activity peptides)\n"
            report += "-"*70 + "\n\n"

            for idx, row in results_df.iterrows():
                report += f"\n{row['Display_Name']}:\n"
                report += f"  Target Range: {row['Target_Range']}\n"
                report += f"  Low Group (Q10): Mean={row['Q10_Group_Mean']}\n"
                report += f"  High Group (Q90): Mean={row['Q90_Group_Mean']}\n"
                report += f"  Current Data: Mean={row['Current_Mean']}, Std={row['Current_Std']}\n"
                report += f"  Current Range: [{row['Current_Min']}, {row['Current_Max']}]\n"
                report += f"  Status: {row['Status']}\n"

            report += "\n" + "="*70 + "\n"

            report_file = output_excel.replace('.xlsx', '_report.txt')
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)

            print(f"\nHydrophobic Face Optimization report saved to: {report_file}")

        print(f"\nHydrophobic Face Optimization results saved to: {output_excel}")

    return results_df


# ==================== Bind MIC-group functions to PeptideAnalyzer class ====================
try:
    PeptideAnalyzer.add_mic_group = add_mic_group
    PeptideAnalyzer._mic_group_summary_one = _mic_group_summary_one
    PeptideAnalyzer._mic_group_tests_one = _mic_group_tests_one
    PeptideAnalyzer.plot_feature_by_mic_group = plot_feature_by_mic_group
    PeptideAnalyzer.run_mic_group_analysis = run_mic_group_analysis
    PeptideAnalyzer.analyze_hydrophobic_face_optimization = analyze_hydrophobic_face_optimization
except Exception:
    pass


def main():
    """Main function: Execute complete analysis workflow"""
    try:
        file_path = get_file_path()

        print("\nInitializing peptide property analyzer...")
        analyzer = PeptideAnalyzer(file_path)
        print(f"Output directory created: {analyzer.output_dir}")

        factor_cols, target_col = analyzer.get_columns_by_selection()

        analyzer.add_delta_hcs_features(verbose=True)
        for derived_col in ["Delta_HCS4", "Delta_HCS3"]:
            if derived_col in analyzer.data.columns and derived_col not in factor_cols:
                factor_cols.append(derived_col)
        print("\n[INFO] Final factor columns include derived Delta HCS features when available.")

        print("\n" + "="*70)
        print("PERFORMING ANALYSIS")
        print("="*70)

        plot_files = analyzer.create_all_individual_plots(factor_cols, target_col)

        # MIC-group stratified analysis (two-class: Low ≤ 3, High > 3)
        mic_outputs = analyzer.run_mic_group_analysis(
            factor_cols=factor_cols,
            target_col=target_col,
            low_th=3.0,  # Two-class threshold
            group_col="MIC_Group",
            top_k=30
        )
        print("\n" + "="*70)
        print("MIC-GROUP STRATIFIED ANALYSIS OUTPUTS (Two-class Low/High, Mann-Whitney U)")
        print("="*70)
        for k, v in mic_outputs.items():
            print(f"{k}: {v}")

        summary_plots = analyzer.create_summary_plots(factor_cols, target_col)

        print("\n" + "="*70)
        print("MACHINE LEARNING ANALYSIS")
        print("="*70)

        print("\nTraining models...")
        model_results = analyzer.train_models(factor_cols, target_col)

        print("\nAnalyzing feature importance...")
        importance_df = analyzer.feature_importance_analysis(factor_cols, target_col)

        # Hydrophobic Face Optimization Analysis (using two-class Low/High)
        print("\n" + "="*70)
        print("HYDROPHOBIC FACE OPTIMIZATION (Q10/Q90 Dynamic Ranges, Two-class MIC)")
        print("="*70)
        print("\nMODIFIED: Target Ranges calculated from Low (MIC ≤ 3) and High (MIC > 3) groups")
        print("  Target Range: [Low_Group_Mean, High_Group_Mean]")

        hf_results = analyzer.analyze_hydrophobic_face_optimization(
            output_excel=os.path.join(analyzer.output_dir, "HF_Optimization_Results.xlsx")
        )

        print("\nHydrophobic Face Optimization Status:")
        print(hf_results[['Display_Name', 'Target_Range', 'Current_Mean', 'Status']].to_string(index=False))

        print("\n" + "="*70)
        print("EXPORTING RESULTS")
        print("="*70)

        output_file = analyzer.export_results(factor_cols, target_col)

        print("\n" + "="*70)
        print("GENERATING FINAL REPORT")
        print("="*70)

        report = analyzer.generate_report(factor_cols, target_col)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE - SUMMARY")
        print("="*70)

        print(f"\nAll results saved to: {analyzer.output_dir}/")
        print("\nGenerated files:")
        print(f"  1. {analyzer.output_dir}/{target_col}_analysis_report.txt - Text analysis report")
        print(f"  2. {analyzer.output_dir}/{target_col}_analysis_report.md - Markdown report")
        print(f"  3. {analyzer.output_dir}/{target_col}_analysis_results.xlsx - Excel results")
        print(f"\nGenerated plots in {analyzer.plot_dir}/:")
        print(f"  4. correlations/ - Individual correlation plots ({len(factor_cols)} files)")
        print(f"  5. interval_analysis/ - Individual interval analysis plots ({len(factor_cols)} files)")
        print(f"  6. model_results/ - Model result plots (4 files)")
        print(f"  7. feature_importance/ - Feature importance plots ({len(factor_cols) + 1} files)")
        print(f"  8. correlation_summary.png - Summary of all correlations")
        print(f"  9. target_distribution.png - Target distribution plot")
        print(f"  10. feature_importance_summary.png - Summary of feature importance")

        print("\n" + "="*70)
        print("Thank you for using the Peptide Property Analysis System!")
        print("="*70)

    except FileNotFoundError as e:
        print(f"\nERROR: File not found: {str(e)}")
        print("Please check the file path and try again.")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check your data format and try again.")


if __name__ == "__main__":
    main()