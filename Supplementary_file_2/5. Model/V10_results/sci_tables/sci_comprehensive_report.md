# SCI Publication-Ready Analysis Report - V10
## Best Model: SelectFromModel-RF-10 + CatBoost_tuned
## Generated: 2026-05-24 00:07:58

---

## 1. Dataset Information

| Property | Value |
|----------|-------|
| Total Samples | 75 |
| Total Features | 14 |
| Selected Features | 9 |
| Classes | High, Low |
| Classification Type | Two-class (Low ≤3, High >3) |

### Class Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| High | 50 | 66.7% |
| Low | 25 | 33.3% |

---

## 2. Best Model Configuration

| Property | Value |
|----------|-------|
| Model | CatBoost_tuned |
| Model Class | CatBoostClassifier |
| Feature Selection | SelectFromModel-RF-10 |
| Number of Features | 10 |

### Selected Features (9 features)
1. Hyd
2. HoF_HMom_PDB
3. HMom_PDB
4. HiF_HiMom_PDB
5. Center_Angle_deg
6. HCS4_HF_Mean_SD_PDB
7. HCS3_HF_Mean_PDB
8. HCS3_HF_Mean_SD_PDB
9. HoF_HiF_Distance

---

## 3. Cross-Validation Results (5-Fold Stratified CV)

### Overall Performance Metrics
| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| Accuracy | 0.8800 ± 0.0267 | [0.8667, 0.9267] |
| Balanced Accuracy | 0.8500 ± 0.0316 | [0.8050, 0.8950] |
| F1 Score (Macro) | 0.8600 ± 0.0313 | [0.8316, 0.9136] |
| F1 Score (Weighted) | 0.8775 ± 0.0272 | [0.8571, 0.9248] |
| Precision (Macro) | 0.8842 ± 0.0436 | [0.8500, 0.9508] |
| Recall (Macro) | 0.8500 ± 0.0316 | [0.8050, 0.8950] |
| ROC-AUC | 0.8800 ± 0.0607 | [0.8020, 0.9560] |
| PR-AUC | 0.8300 ± 0.1072 | [0.6576, 0.9397] |
| Log Loss | 0.4506 ± 0.1690 | [0.2898, 0.7332] |
| MCC | 0.7320 ± 0.0605 | [0.7000, 0.8382] |
| Cohen's Kappa | 0.7218 ± 0.0615 | [0.6700, 0.8279] |

### Per-Class Performance
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| High | 0.8885 ± 0.0278 | 0.9400 ± 0.0490 | 0.9123 ± 0.0204 |
| Low | 0.8800 ± 0.0980 | 0.7600 ± 0.0800 | 0.8078 ± 0.0449 |

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
1. **Hyd** (Importance: 0.9940, Rank: 1.2)
2. **HCS3_HF_Mean_PDB** (Importance: 0.3384, Rank: 5.0)
3. **HCS3_HF_Mean_SD_PDB** (Importance: 0.2588, Rank: 4.7)
4. **HiF_HiMom_PDB** (Importance: 0.2183, Rank: 4.5)
5. **HCS4_HF_Mean_SD_PDB** (Importance: 0.1703, Rank: 5.5)
6. **Center_Angle_deg** (Importance: 0.1647, Rank: 4.1)
7. **HoF_HiF_Distance** (Importance: 0.1554, Rank: 7.3)
8. **HoF_HMom_PDB** (Importance: 0.1233, Rank: 6.5)
9. **HMom_PDB** (Importance: 0.1129, Rank: 6.2)

---

## 5. Permutation Importance Analysis

Statistical significance testing was performed using 30 permutation repeats.

**Significant Features (p<0.05):** 1/9

| Feature | Importance | Std | P-value | Significant |
|---------|------------|-----|--------|-------------|
| Hyd | 0.1604 | 0.0427 | 3.3492e-01 | ✗ |
| Center_Angle_deg | 0.0298 | 0.0203 | 4.3111e-02 | ✓ |
| HiF_HiMom_PDB | 0.0276 | 0.0142 | 1.9258e-01 | ✗ |
| HCS3_HF_Mean_PDB | 0.0142 | 0.0198 | 3.6126e-01 | ✗ |
| HCS4_HF_Mean_SD_PDB | 0.0080 | 0.0148 | 6.9932e-02 | ✗ |
| HMom_PDB | 0.0013 | 0.0087 | 2.2171e-01 | ✗ |
| HoF_HiF_Distance | -0.0027 | 0.0080 | 3.2754e-01 | ✗ |
| HCS3_HF_Mean_SD_PDB | -0.0031 | 0.0082 | 1.5055e-01 | ✗ |
| HoF_HMom_PDB | -0.0049 | 0.0064 | 1.6700e-01 | ✗ |

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

1. **Best Performing Model**: CatBoost_tuned with SelectFromModel-RF-10
2. **Expected Performance**: Accuracy = 0.8800, F1 = 0.8600
3. **Most Important Features**: Top 3 features identified by multiple methods

### Clinical/Biological Interpretation

- Features with highest importance contribute most to High vs Low classification
- The model can potentially be used to predict MIC category from peptide features

---

## 9. Statistical Methods Used

1. **Cross-Validation**: 5-Fold Stratified Cross-Validation
2. **Feature Importance**: Multiple methods (MI, ANOVA, RF, XGB, Correlation)
3. **Statistical Tests**: Permutation tests with t-test for significance
4. **Confidence Intervals**: 95% CI calculated from CV folds
5. **Stability Analysis**: 10 repeated CV runs

---

## 10. Reproducibility

- Random Seed: 42 (used throughout the analysis)
- Software Version: V10
- Date Generated: 2026-05-24 00:07:58

---

**Report Generated by AMP Modeling V10 - SCI Publication Analysis Module**
