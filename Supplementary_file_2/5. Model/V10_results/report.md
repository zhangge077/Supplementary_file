# AMP MIC Prediction Model Report - V10
## Enhanced Version with LOOCV, Feature Selection, LDA SHAP + SCI Analysis
## Classification Type: Two-Class (Low, High)

## Classification Thresholds
| Category | Exp_Log2_MIC Range | Description |
|----------|-------------------|-------------|
| **Low** | <= 3 | Sensitive (lower MIC indicates better activity) |
| **High** | > 3 | Less sensitive/Resistant (higher MIC indicates reduced activity) |

## Dataset Information
- **Number of samples**: 75
- **Number of features**: 14
- **Number of classes**: 2 (Two-class)
- **Classes**: High, Low

## Class Distribution
| Category | Threshold | Count | Percentage |
|----------|-----------|-------|------------|
| High | >3 | 50 | 66.7% |
| Low | <=3 | 25 | 33.3% |

## Feature Selection Methods (V10)
- **SelectKBest (f_classif)**: 5, 10 features
- **SelectKBest (mutual_info)**: 5, 10 features
- **SelectFromModel (RandomForest)**: 5, 10 features
- **SelectFromModel (XGBoost)**: 5, 10 features
- **PCA**: 5, 10 components
- **No Reduction**: All features

## Best Model Configuration
- **Reduction Method**: SelectFromModel-RF-10
- **Model**: CatBoost_tuned
- **Model Class**: CatBoostClassifier
- **Number of Selected Features**: 10

## All Original Features (for Scaler): 14
Length, Hyd, z, FreqPolar, FreqNonPolar, HoF_HMom_PDB, HMom_PDB, HiF_HiMom_PDB, Center_Angle_deg, HCS4_HF_Mean_PDB, HCS4_HF_Mean_SD_PDB, HCS3_HF_Mean_PDB, HCS3_HF_Mean_SD_PDB, HoF_HiF_Distance


## Selected Features (for Model): 9
Hyd, HoF_HMom_PDB, HMom_PDB, HiF_HiMom_PDB, Center_Angle_deg, HCS4_HF_Mean_SD_PDB, HCS3_HF_Mean_PDB, HCS3_HF_Mean_SD_PDB, HoF_HiF_Distance

## Cross-Validation Results (5-Fold)
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8800 ± 0.0267 |
| **F1 (Macro)** | 0.8600 |
| **F1 (Weighted)** | 0.8775 |
| **Precision (Macro)** | 0.8842 |
| **Recall (Macro)** | 0.8500 |
| **ROC-AUC** | 0.8800 |

## Test Set Results
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8000 |
| **F1 (Macro)** | 0.7847 |
| **ROC-AUC** | 0.9000 |
| **Balanced Accuracy** | 0.8000 |
| **MCC** | 0.5774 |
| **Kappa** | 0.5714 |

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
**Generated**: 2026-05-24 00:07:58
