#!/usr/bin/env python3
"""
MIC Prediction Program - V9 (Revised for Two-Class Classification)
Predict MIC category (Low/High) for candidate peptides based on trained V9 model

CRITICAL FIX: This version correctly handles the feature selection pipeline:
1. Load ALL original features from input (e.g., 54 features)
2. Standardize using scaler (trained on all 54 features)
3. Apply feature selector to reduce to selected features (e.g., 10 features)
4. Predict using the trained model (Two-class: Low, High)

Classification Thresholds (Based on Exp_Log2_MIC):
    - Low (<=3): Sensitive (lower MIC indicates better activity)
    - High (>3): Less sensitive/Resistant (higher MIC indicates reduced activity)

Usage:
    python MIC_Predicted_V10.py --input INPUT.xlsx --output OUTPUT.xlsx

    Or place input.xlsx in the same directory as this script.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Default paths
DEFAULT_MODEL_DIR = 'V10_results'
DEFAULT_INPUT_FILE = 'KAWL_Model_Data.xlsx'
DEFAULT_OUTPUT_FILE = 'KAWL_Model_Data.xlsx'


class MICPredictorV9:
    """MIC Prediction Class based on V9 trained model - Two-Class Classification"""

    def __init__(self, model_dir=None):
        """Initialize predictor with model directory"""
        if model_dir is None:
            # Try to find model directory
            possible_paths = [
                DEFAULT_MODEL_DIR,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_MODEL_DIR),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DEFAULT_MODEL_DIR),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    self.model_dir = path
                    break
            else:
                self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_MODEL_DIR)
        else:
            self.model_dir = model_dir

        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_selector = None
        self.model_info = None
        self.all_feature_names = None  # ALL original features (e.g., 54)
        self.selected_feature_names = None  # Selected features after feature selection (e.g., 10)
        self.class_names = None

    def load_model(self):
        """Load trained model and related files"""
        print("=" * 80)
        print("LOADING TRAINED MODEL (V9 - Two-Class)")
        print("=" * 80)
        print(f"\nModel directory: {self.model_dir}")

        # Check if model directory exists
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        # Load model info
        model_info_path = os.path.join(self.model_dir, 'model_info.json')
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                self.model_info = json.load(f)
            print(f"\n[Model Information]")
            print(f"  Model Class: {self.model_info.get('model_class', 'Unknown')}")
            print(f"  Model Description: {self.model_info.get('description', 'Unknown')}")
            print(f"  Reduction Method: {self.model_info.get('reduction_method', 'Unknown')}")
            cv_acc = self.model_info.get('cv_accuracy')
            cv_f1 = self.model_info.get('cv_f1_macro')
            cv_auc = self.model_info.get('cv_roc_auc')
            print(f"  CV Accuracy: {f'{cv_acc:.4f}' if isinstance(cv_acc, float) else 'N/A'}")
            print(f"  CV F1 Macro: {f'{cv_f1:.4f}' if isinstance(cv_f1, float) else 'N/A'}")
            print(f"  CV ROC-AUC: {f'{cv_auc:.4f}' if isinstance(cv_auc, float) else 'N/A'}")
            print(f"  Classification Type: {self.model_info.get('classification_type', 'two-class')}")
            print(f"  Total Features (for Scaler): {self.model_info.get('n_all_features', 'Unknown')}")
            print(f"  Selected Features (for Model): {self.model_info.get('n_selected_features', 'Unknown')}")

            # Show classification thresholds
            thresholds = self.model_info.get('thresholds', {})
            if thresholds:
                print(f"\n[Classification Thresholds]")
                for cat, thresh in thresholds.items():
                    print(f"  {cat}: {thresh}")
        else:
            raise FileNotFoundError(f"model_info.json not found in {self.model_dir}")

        # Load the trained model
        model_path = os.path.join(self.model_dir, 'best_model.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"\n[Model loaded successfully]")
        else:
            raise FileNotFoundError(f"best_model.pkl not found in {self.model_dir}")

        # Load scaler
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"[Scaler loaded successfully]")
        else:
            raise FileNotFoundError(f"scaler.pkl not found in {self.model_dir}")

        # Load label encoder
        label_encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
        if os.path.exists(label_encoder_path):
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            self.class_names = list(self.label_encoder.classes_)
            print(f"[Label encoder loaded successfully]")
            print(f"  Classes: {self.class_names}")
        else:
            raise FileNotFoundError(f"label_encoder.pkl not found in {self.model_dir}")

        # Load feature selector (if exists)
        feature_selector_path = os.path.join(self.model_dir, 'feature_selector.pkl')
        if os.path.exists(feature_selector_path):
            with open(feature_selector_path, 'rb') as f:
                self.feature_selector = pickle.load(f)
            print(f"[Feature selector loaded successfully]")
        else:
            print(f"[Note: No feature_selector.pkl found, using all scaled features]")

        # Get ALL original feature names (needed for scaler)
        self.all_feature_names = self.model_info.get('all_feature_names', [])

        if not self.all_feature_names:
            # Try to load from tables/all_original_features.csv
            all_features_path = os.path.join(self.model_dir, 'tables', 'all_original_features.csv')
            if os.path.exists(all_features_path):
                all_features_df = pd.read_csv(all_features_path)
                self.all_feature_names = all_features_df['Feature'].tolist()
            else:
                self.all_feature_names = self.model_info.get('selected_features', [])

        # Get selected feature names (for model input)
        self.selected_feature_names = self.model_info.get('selected_features', [])

        if not self.selected_feature_names:
            selected_features_path = os.path.join(self.model_dir, 'tables', 'selected_features.csv')
            if os.path.exists(selected_features_path):
                selected_df = pd.read_csv(selected_features_path)
                self.selected_feature_names = selected_df['Feature'].tolist()

        print(f"\n[Feature Configuration]")
        print(f"  ALL original features (for scaler): {len(self.all_feature_names)}")
        print(f"  Selected features (for model): {len(self.selected_feature_names)}")

        if self.selected_feature_names:
            print(f"\n  Selected features:")
            for i, feat in enumerate(self.selected_feature_names, 1):
                print(f"    {i}. {feat}")

        print("\n" + "=" * 80)

        return self

    def preprocess_input_data(self, df):
        """
        CRITICAL: Preprocess input data with correct feature pipeline for Two-Class:

        Step 1: Extract ALL original features from input (matching training data)
        Step 2: Fill missing values with median from training
        Step 3: Standardize using scaler (trained on ALL original features)
        Step 4: Apply feature selector to reduce to selected features
        Step 5: Return preprocessed data for model prediction (Two-class)
        """
        print("\n[Preprocessing Input Data]")
        print("  Pipeline: ALL features -> Scaler -> Feature Selector -> Model")
        print("  Classification: Two-class (Low, High)")

        # Step 1: Get ALL original feature columns from input
        exclude_cols = ['Name', 'Sequences', 'Exp_Log2_MIC', 'MIC_Category', 'Predicted_MIC_Category']
        input_numeric_cols = [col for col in df.columns
                             if col not in exclude_cols and df[col].dtype in ['int64', 'float64']]

        print(f"\n  Step 1: Loading ALL original features")
        print(f"    Input columns: {len(df.columns)}")
        print(f"    Numeric columns in input: {len(input_numeric_cols)}")
        print(f"    Expected features for scaler: {len(self.all_feature_names)}")

        # Check which ALL original features are available in input
        missing_all_features = []
        available_all_features = []
        for feat in self.all_feature_names:
            if feat in df.columns:
                available_all_features.append(feat)
            else:
                missing_all_features.append(feat)

        if missing_all_features:
            print(f"\n  Warning: {len(missing_all_features)} features not found in input data")
            print(f"    Missing features: {missing_all_features[:10]}...")
            if len(missing_all_features) > 10:
                print(f"    ... and {len(missing_all_features) - 10} more")

        # Use only available features that match training
        X_all = df[available_all_features].copy()

        # Step 2: Handle missing values
        print(f"\n  Step 2: Handling missing values")
        missing_count = X_all.isnull().sum().sum()
        if missing_count > 0:
            print(f"    Filling {missing_count} missing values with median")
            X_all = X_all.fillna(X_all.median())

        # Verify feature count matches scaler expectation
        expected_n_features = len(self.all_feature_names)
        actual_n_features = X_all.shape[1]

        if actual_n_features != expected_n_features:
            print(f"\n  ERROR: Feature count mismatch!")
            print(f"    Expected (for scaler): {expected_n_features}")
            print(f"    Available in input: {actual_n_features}")
            print(f"    Please ensure input file has all {expected_n_features} features")
            raise ValueError(f"Feature count mismatch: expected {expected_n_features}, got {actual_n_features}")

        # Step 3: Standardize features using scaler
        print(f"\n  Step 3: Standardizing features (using scaler on {X_all.shape[1]} features)")
        X_scaled = self.scaler.transform(X_all.values)
        print(f"    Scaled data shape: {X_scaled.shape}")

        # Step 4: Apply feature selection (if available)
        print(f"\n  Step 4: Applying feature selection")
        if self.feature_selector is not None:
            X_final = self.feature_selector.transform(X_scaled)
            print(f"    After feature selection: {X_final.shape[1]} features")
        else:
            X_final = X_scaled
            print(f"    No feature selector applied, using all {X_final.shape[1]} features")

        # Verify final feature count matches model expectation
        expected_final_features = len(self.selected_feature_names) if self.selected_feature_names else X_scaled.shape[1]
        if X_final.shape[1] != expected_final_features:
            print(f"\n  Warning: Feature count after selection ({X_final.shape[1]}) != model expectation ({expected_final_features})")

        print(f"\n  Preprocessing complete!")
        print(f"    Final shape: {X_final.shape}")

        # Get sample names
        sample_names = df['Name'] if 'Name' in df.columns else pd.Series([f'Peptide_{i}' for i in range(len(df))])

        return X_final, sample_names, X_all

    def predict(self, X):
        """Make predictions on preprocessed data"""
        predictions = self.model.predict(X)
        pred_labels = self.label_encoder.inverse_transform(predictions)

        # Get probability predictions if available
        pred_proba = None
        if hasattr(self.model, 'predict_proba'):
            pred_proba = self.model.predict_proba(X)

        return pred_labels, pred_proba

    def predict_from_file(self, input_path, output_path=None):
        """Predict MIC categories from input Excel file - Two-Class"""
        print("\n" + "=" * 80)
        print("MIC PREDICTION (Two-Class)")
        print("=" * 80)

        # Load input data
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        print(f"\nLoading input file: {input_path}")
        df_input = pd.read_excel(input_path)
        print(f"  Total samples: {len(df_input)}")
        print(f"  Input columns: {list(df_input.columns)}")

        # Preprocess
        X_preprocessed, sample_names, X_all_features = self.preprocess_input_data(df_input)

        # Predict
        print("\n[Making Predictions - Two-Class]")
        pred_labels, pred_proba = self.predict(X_preprocessed)

        # Extract sequences from input
        sequences = df_input['Sequences'] if 'Sequences' in df_input.columns else pd.Series([''] * len(df_input))

        # Create results dataframe
        results = pd.DataFrame({
            'Name': sample_names.values,
            'Sequences': sequences.values,
            'Predicted_MIC_Category': pred_labels,
        })

        # Add probability information if available
        if pred_proba is not None:
            for i, cls in enumerate(self.class_names):
                results[f'Probability_{cls}'] = pred_proba[:, i]

            # Calculate confidence score
            max_proba = np.max(pred_proba, axis=1)
            confidence_labels = []
            for p in max_proba:
                if p >= 0.8:
                    confidence_labels.append('High')
                elif p >= 0.6:
                    confidence_labels.append('Medium')
                else:
                    confidence_labels.append('Low')
            results['Prediction_Confidence'] = confidence_labels
            results['Max_Probability'] = max_proba

        # Add prediction interpretation for two classes
        def interpret_category(x):
            if x == 'Low':
                return 'Low MIC (Sensitive - Better Activity)'
            else:
                return 'High MIC (Less Sensitive - Reduced Activity)'

        results['Interpretation'] = results['Predicted_MIC_Category'].apply(interpret_category)

        # Summary
        print("\n[Prediction Summary - Two Classes]")
        pred_dist = results['Predicted_MIC_Category'].value_counts()
        for cat in ['Low', 'High']:
            count = pred_dist.get(cat, 0)
            pct = count / len(results) * 100 if len(results) > 0 else 0
            print(f"  {cat:8s}: {count:3d} samples ({pct:5.1f}%)")

        # Save results
        if output_path is None:
            output_path = os.path.join(os.path.dirname(input_path) if input_path else '.', DEFAULT_OUTPUT_FILE)

        # Create detailed results with all features
        detailed_results = results.copy()

        # Add ALL original features for reference
        for col in X_all_features.columns:
            detailed_results[col] = X_all_features[col].values

        # Add selected features indicator
        if self.selected_feature_names:
            detailed_results['Selected_Features'] = ', '.join(self.selected_feature_names[:5])
            if len(self.selected_feature_names) > 5:
                detailed_results['Selected_Features'] += f'... (+{len(self.selected_feature_names) - 5} more)'

        # Save results to Excel with multiple sheets
        print(f"\n[Saving Results]")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            results.to_excel(writer, sheet_name='Predictions_Summary', index=False)
            detailed_results.to_excel(writer, sheet_name='Detailed_Results', index=False)

            # Save model info sheet
            cv_acc = self.model_info.get('cv_accuracy', 'N/A')
            cv_f1 = self.model_info.get('cv_f1_macro', 'N/A')
            cv_auc = self.model_info.get('cv_roc_auc', 'N/A')
            thresholds = self.model_info.get('thresholds', {})
            model_info_df = pd.DataFrame([
                ['Model Class', self.model_info.get('model_class', 'N/A')],
                ['Model Description', self.model_info.get('description', 'N/A')],
                ['Reduction Method', self.model_info.get('reduction_method', 'N/A')],
                ['Classification Type', self.model_info.get('classification_type', 'two-class')],
                ['CV Accuracy', f"{cv_acc:.4f}" if isinstance(cv_acc, float) else 'N/A'],
                ['CV F1 Macro', f"{cv_f1:.4f}" if isinstance(cv_f1, float) else 'N/A'],
                ['CV ROC-AUC', f"{cv_auc:.4f}" if isinstance(cv_auc, float) else 'N/A'],
                ['Total Features (Scaler)', self.model_info.get('n_all_features', 'N/A')],
                ['Selected Features (Model)', self.model_info.get('n_selected_features', 'N/A')],
                ['Classes', ', '.join(self.class_names)],
                ['Threshold Low', thresholds.get('Low', '<=3')],
                ['Threshold High', thresholds.get('High', '>3')],
                ['Prediction Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ], columns=['Property', 'Value'])
            model_info_df.to_excel(writer, sheet_name='Model_Info', index=False)

        print(f"\n[Results saved to: {output_path}]")

        return results

    def batch_predict_from_directory(self, input_dir, output_dir=None):
        """Batch predict from all Excel files in a directory - Two-Class"""
        print("\n" + "=" * 80)
        print("BATCH PREDICTION (Two-Class)")
        print("=" * 80)

        if output_dir is None:
            output_dir = os.path.join(self.model_dir, 'batch_predictions')
        os.makedirs(output_dir, exist_ok=True)

        # Find all Excel files
        excel_files = [f for f in os.listdir(input_dir)
                      if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]

        print(f"\nFound {len(excel_files)} Excel files to process")

        all_results = []

        for i, filename in enumerate(excel_files, 1):
            input_path = os.path.join(input_dir, filename)
            output_filename = f"predictions_{filename}"
            output_path = os.path.join(output_dir, output_filename)

            print(f"\n[{i}/{len(excel_files)}] Processing: {filename}")

            try:
                results = self.predict_from_file(input_path, output_path)

                # Count two classes
                pred_dist = results['Predicted_MIC_Category'].value_counts()
                all_results.append({
                    'filename': filename,
                    'n_samples': len(results),
                    'n_low': pred_dist.get('Low', 0),
                    'n_high': pred_dist.get('High', 0),
                    'status': 'Success',
                    'output_file': output_path
                })
            except Exception as e:
                all_results.append({
                    'filename': filename,
                    'n_samples': 0,
                    'n_low': 0,
                    'n_high': 0,
                    'status': f'Error: {str(e)}',
                    'output_file': None
                })
                print(f"  Error: {str(e)}")

        # Save batch summary
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(output_dir, 'batch_prediction_summary.xlsx')
        summary_df.to_excel(summary_path, index=False)

        print(f"\n" + "=" * 80)
        print("BATCH PREDICTION COMPLETE (Two-Class)")
        print("=" * 80)
        print(f"Processed: {len(all_results)} files")
        print(f"Successful: {sum(1 for r in all_results if r['status'] == 'Success')}")
        print(f"Failed: {sum(1 for r in all_results if r['status'] != 'Success')}")
        print(f"Results saved to: {output_dir}")

        return summary_df


def create_sample_input_template(output_path='sample_input_template.xlsx'):
    """Create a sample input template based on actual model features - Two-Class"""
    print("\n" + "=" * 80)
    print("CREATING SAMPLE INPUT TEMPLATE (Two-Class)")
    print("=" * 80)

    # Try to load actual features from model
    model_dir = DEFAULT_MODEL_DIR
    model_info_path = os.path.join(model_dir, 'model_info.json')

    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)

        all_features = model_info.get('all_feature_names', [])
        print(f"\nFound {len(all_features)} features from V9 model")

        thresholds = model_info.get('thresholds', {})
        if thresholds:
            print(f"\nClassification Thresholds:")
            for cat, thresh in thresholds.items():
                print(f"  {cat}: {thresh}")
    else:
        # Use common peptide features
        all_features = [
            'Length', 'Molecular_Weight', 'Charge', 'pI', 'Hyd', 'HMom',
            'GRAVY', 'Hydrophobicity', 'Hydrophobic_Moment', 'z',
            'FreqPolar', 'FreqNonPolar', 'Freq_A', 'Freq_C', 'Freq_D', 'Freq_E',
            'Freq_F', 'Freq_G', 'Freq_H', 'Freq_I', 'Freq_K', 'Freq_L', 'Freq_M',
            'Freq_N', 'Freq_P', 'Freq_Q', 'Freq_R', 'Freq_S', 'Freq_T', 'Freq_V',
            'Freq_W', 'Freq_Y', 'Freq_Other', 'AlphaHelix', 'BetaSheet',
            'Turn', 'RandomCoil', 'Instability_Index', 'Aliphatic_Index',
            'Boman_Index', 'Membrane_Location', 'Solubility', 'Charge_Density',
            'Net_Hydrophobicity', 'Hydrophobic_Ratio', 'Charge_Ratio',
            'Polar_AA_Ratio', 'Aromatic_AA_Ratio', 'Tiny_AA_Ratio',
            'Small_AA_Ratio', 'Large_AA_Ratio', 'pI_Normalized', 'Charge_pI_Ratio'
        ]
        print(f"\nUsing {len(all_features)} common peptide features")
        print("\nClassification Thresholds (Two-Class):")
        print("  Low: <=3")
        print("  High: >3")

    template = pd.DataFrame()
    template['Name'] = ['Peptide_1', 'Peptide_2', 'Peptide_3', 'Peptide_4', 'Peptide_5']
    template['Sequences'] = ['AGHKHLPSLGKK', 'AAKKFFRRGG', 'RRLPRFFKKG', 'VKLKHKAILSKKK', 'LLPWLREKG']

    # Add all features with NaN (placeholder values)
    for feat in all_features:
        if feat not in template.columns:
            template[feat] = [np.nan] * 5

    template.to_excel(output_path, index=False)
    print(f"\nSample template created: {output_path}")
    print(f"Please fill in the feature values for your candidate peptides.")
    print(f"Required features: {len(all_features)}")


def validate_input_file(input_path):
    """Validate input file has required features"""
    print("\n[Validating Input File]")
    model_dir = DEFAULT_MODEL_DIR
    model_info_path = os.path.join(model_dir, 'model_info.json')

    if not os.path.exists(model_info_path):
        print(f"  Warning: model_info.json not found in {model_dir}")
        return True

    with open(model_info_path, 'r') as f:
        model_info = json.load(f)

    required_features = model_info.get('all_feature_names', [])
    if not required_features:
        print(f"  Warning: No features found in model_info")
        return True

    # Show classification info
    classification_type = model_info.get('classification_type', 'two-class')
    print(f"  Model classification type: {classification_type}")

    thresholds = model_info.get('thresholds', {})
    if thresholds:
        print(f"  Classification thresholds:")
        for cat, thresh in thresholds.items():
            print(f"    {cat}: {thresh}")

    df = pd.read_excel(input_path)
    available_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

    missing = set(required_features) - set(available_features)
    extra = set(available_features) - set(required_features)

    print(f"\n  Required features: {len(required_features)}")
    print(f"  Available features: {len(available_features)}")

    if missing:
        print(f"\n  Missing features ({len(missing)}):")
        for feat in list(missing)[:10]:
            print(f"    - {feat}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if extra:
        print(f"\n  Extra features (will be ignored, {len(extra)}):")
        for feat in list(extra)[:10]:
            print(f"    + {feat}")

    return len(missing) == 0


def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("MIC PREDICTION PROGRAM - V9 (Two-Class Classification)")
    print("Predict MIC categories for candidate antimicrobial peptides")
    print("Classification: Low (<=3), High (>3)")
    print("=" * 80)

    # Parse arguments
    parser = argparse.ArgumentParser(description='MIC Prediction for Peptides (Two-Class)')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='Input Excel file path (default: candidate_peptides.xlsx)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output Excel file path')
    parser.add_argument('--model_dir', '-m', type=str, default=None,
                       help='Model directory (default: V9_results)')
    parser.add_argument('--batch', '-b', type=str, default=None,
                       help='Batch mode: directory containing multiple Excel files')
    parser.add_argument('--template', '-t', action='store_true',
                       help='Create sample input template')
    parser.add_argument('--validate', '-v', type=str, default=None,
                       help='Validate input file has required features')

    args = parser.parse_args()

    # Handle template creation
    if args.template:
        create_sample_input_template()
        return

    # Handle validation
    if args.validate:
        validate_input_file(args.validate)
        return

    # Initialize predictor
    try:
        predictor = MICPredictorV9(model_dir=args.model_dir)
        predictor.load_model()
    except Exception as e:
        print(f"\nError loading model: {str(e)}")
        print("\nPlease ensure:")
        print("  1. The V9_results directory exists")
        print("  2. It contains: best_model.pkl, scaler.pkl, label_encoder.pkl, model_info.json")
        print("  3. Run amp_modeling_enum_V9.py first to train the model")
        return

    # Batch mode
    if args.batch:
        predictor.batch_predict_from_directory(args.batch)
        return

    # Single file mode
    input_file = args.input
    if input_file is None:
        # Try default locations
        possible_paths = [
            DEFAULT_INPUT_FILE,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_INPUT_FILE),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', DEFAULT_INPUT_FILE),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                input_file = path
                break

    if input_file is None or not os.path.exists(input_file):
        print(f"\nInput file not specified or not found.")
        print(f"\nPlease provide input file using --input:")
        print(f"  python MIC_Predicted_V9.py --input candidate_peptides.xlsx")

        print("\nTo create a sample template, run with --template flag:")
        print("  python MIC_Predicted_V9.py --template")

        print("\nTo validate your input file, run with --validate:")
        print("  python MIC_Predicted_V9.py --validate your_file.xlsx")

        return

    output_file = args.output

    # Optional: validate input file
    if os.path.exists(input_file):
        try:
            validate_input_file(input_file)
        except Exception as e:
            print(f"\nWarning: Validation failed: {str(e)}")

    try:
        results = predictor.predict_from_file(input_file, output_file)

        print("\n" + "=" * 80)
        print("PREDICTION COMPLETE! (Two-Class Classification)")
        print("=" * 80)
        print(f"\nResults Summary:")
        print(f"  Total predictions: {len(results)}")
        print(f"  Output file: {output_file or 'mic_predictions_results.xlsx'}")
        print(f"\nClassification categories:")
        print(f"  Low (<=3): Sensitive - Better Activity")
        print(f"  High (>3): Less Sensitive - Reduced Activity")

    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
