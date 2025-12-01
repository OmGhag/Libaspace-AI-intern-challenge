"""
Automated Form Type Classification Pipeline
Complete end-to-end workflow from raw form data to final submission

This version handles multiple JSON formats automatically.

Usage:
    python automated_pipeline_fixed.py --raw-form raw_form.json --standard-form standard_form.json
"""

import pandas as pd
import json
import numpy as np
import re
import os
import sys
import argparse
from datetime import datetime

# ML imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'low_confidence_threshold': 0.85,
    'high_confidence_threshold': 0.95,
    'num_few_shot_examples': 5,
    'cv_folds': 5,
    'output_dir': '.',
}


# ============================================================================
# DATA LOADING - HANDLES MULTIPLE FORMATS
# ============================================================================

def load_json_data(filepath):
    """Load JSON data from file - handles various formats"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"[OK] Loaded: {filepath}")
        return data
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON: {filepath}")
        return None


def normalize_json_data(data):
    """
    Normalize various JSON formats to list of field objects
    
    Handles:
    - List of objects (standard)
    - Dict of objects (keyed by ID)
    - CSV-like nested structure
    """
    
    if isinstance(data, list):
        # Already a list - check if it's strings or objects
        if len(data) > 0:
            if isinstance(data[0], str):
                print("[INFO] JSON is list of strings - attempting to parse...")
                # Try to parse as JSON strings
                try:
                    parsed = []
                    for item in data:
                        if isinstance(item, str):
                            parsed.append(json.loads(item))
                        else:
                            parsed.append(item)
                    return parsed
                except:
                    print("[ERROR] Could not parse JSON strings")
                    return []
            else:
                # List of objects - good!
                return data
        else:
            return []
    
    elif isinstance(data, dict):
        # Dict of objects - convert to list
        if len(data) > 0:
            first_key = list(data.keys())[0]
            first_val = data[first_key]
            
            if isinstance(first_val, dict):
                # Dict of dicts - convert to list
                result = []
                for key, val in data.items():
                    if isinstance(val, dict):
                        val['id'] = key
                        result.append(val)
                return result
            elif isinstance(first_val, list):
                # Nested structure - flatten
                result = []
                for key, items in data.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict):
                                item['id'] = key
                                result.append(item)
                return result
        return []
    
    print("[WARNING] Unknown JSON format")
    return []


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(field_data):
    """
    Extract features from a form field
    Handles missing fields gracefully
    """
    
    if not isinstance(field_data, dict):
        return None
    
    # Initialize features with defaults
    features = {
        'field_id': str(field_data.get('id', field_data.get('field_id', 'unknown'))),
        'tag': str(field_data.get('tag', '')),
        'type': str(field_data.get('type', '')),
        'has_options': 0,
        'options_count': 0,
        'is_yes_no_question': 0,
        'is_required': 1 if field_data.get('required') else 0,
        'label_length': 0,
        'keyword_select': 0,
    }
    
    # Get label and check length
    label = str(field_data.get('label', ''))
    features['label_length'] = len(label)
    
    # Check for options
    options = field_data.get('options')
    if options:
        if isinstance(options, list):
            features['has_options'] = 1
            features['options_count'] = len(options)
        elif isinstance(options, dict):
            features['has_options'] = 1
            features['options_count'] = len(options)
    
    # Check for yes/no question
    label_lower = label.lower()
    yes_no_keywords = ['yes', 'no', 'are you', 'do you', 'have you', 'will you', 
                       'can you', 'would you', 'should you', 'must you']
    if any(keyword in label_lower for keyword in yes_no_keywords):
        features['is_yes_no_question'] = 1
    
    # Check for select keyword
    features['keyword_select'] = 1 if 'select' in label_lower else 0
    
    return features


def process_raw_form(raw_form_data):
    """Process raw form data - extract features"""
    
    print("\n" + "="*70)
    print("PHASE 1: PROCESSING RAW FORM DATA")
    print("="*70)
    
    # Normalize data format
    normalized = normalize_json_data(raw_form_data)
    
    if len(normalized) == 0:
        print("[ERROR] No valid fields found in raw form")
        return pd.DataFrame()
    
    features_list = []
    for field in normalized:
        try:
            features = extract_features(field)
            if features:
                features_list.append(features)
        except Exception as e:
            print(f"[WARNING] Error processing field: {e}")
            continue
    
    if len(features_list) == 0:
        print("[ERROR] Could not extract features from any fields")
        return pd.DataFrame()
    
    raw_df = pd.DataFrame(features_list)
    
    print(f"\n[OK] Extracted features from {len(raw_df)} fields")
    print(f"   Columns: {list(raw_df.columns)}")
    
    return raw_df


def process_standard_form(standard_form_data):
    """Process standard form data - extract features and labels"""
    
    print("\n" + "="*70)
    print("PHASE 1: PROCESSING STANDARD FORM DATA (Ground Truth)")
    print("="*70)
    
    # Normalize data format
    normalized = normalize_json_data(standard_form_data)
    
    if len(normalized) == 0:
        print("[ERROR] No valid fields found in standard form")
        return pd.DataFrame()
    
    features_list = []
    for field in normalized:
        try:
            features = extract_features(field)
            if features:
                # Add label from standard form
                features['true_kind'] = str(field.get('input_kind', 'text')).lower()
                features_list.append(features)
        except Exception as e:
            print(f"[WARNING] Error processing field: {e}")
            continue
    
    if len(features_list) == 0:
        print("[ERROR] Could not extract features from any fields")
        return pd.DataFrame()
    
    standard_df = pd.DataFrame(features_list)
    
    print(f"\n[OK] Extracted features from {len(standard_df)} fields")
    if 'true_kind' in standard_df.columns:
        print(f"   Text fields: {(standard_df['true_kind'] == 'text').sum()}")
        print(f"   Select fields: {(standard_df['true_kind'] == 'select').sum()}")
    
    return standard_df


# ============================================================================
# PHASE 2: RULE-BASED BASELINE
# ============================================================================

def apply_rule_engine(df):
    """Apply rule-based classification"""
    
    print("\n" + "="*70)
    print("PHASE 2: RULE-BASED CLASSIFICATION (Baseline)")
    print("="*70)
    
    predictions = []
    
    for idx, row in df.iterrows():
        # Rule 1: If has options -> SELECT
        if row['has_options'] == 1 or row['options_count'] > 0:
            predictions.append('select')
        # Rule 2: If yes/no question -> SELECT
        elif row['is_yes_no_question'] == 1:
            predictions.append('select')
        # Rule 3: Otherwise -> TEXT
        else:
            predictions.append('text')
    
    df['rule_prediction'] = predictions
    
    if 'true_kind' in df.columns:
        df['rule_correct'] = df['rule_prediction'] == df['true_kind']
        accuracy = df['rule_correct'].mean()
        print(f"\n[OK] Rule-based accuracy: {accuracy*100:.1f}%")
    
    return df


# ============================================================================
# PHASE 3: LOGISTIC REGRESSION
# ============================================================================

def train_logistic_regression(standard_df):
    """Train Logistic Regression on standard form"""
    
    print("\n" + "="*70)
    print("PHASE 3: TRAINING LOGISTIC REGRESSION")
    print("="*70)
    
    # Features for ML
    feature_cols = ['options_count', 'has_options', 'is_yes_no_question', 
                    'is_required', 'label_length']
    
    X = standard_df[feature_cols].fillna(0)
    y = (standard_df['true_kind'] == 'select').astype(int)
    
    print(f"\nTraining data: {len(X)} samples")
    print(f"Features: {feature_cols}")
    print(f"Label distribution: {y.sum()} select, {(1-y).sum()} text")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_scaled, y)
    
    print(f"\n[OK] Model trained")
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=CONFIG['cv_folds'], shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=cv)
    
    print(f"   5-fold CV accuracy: {cv_scores.mean()*100:.1f}% (+-{cv_scores.std()*100:.1f}%)")
    print(f"   Training accuracy: {model.score(X_scaled, y)*100:.1f}%")
    
    return model, scaler, feature_cols


def predict_with_logistic_regression(df, model, scaler, feature_cols):
    """Predict using trained Logistic Regression"""
    
    print("\n" + "="*70)
    print("PHASE 3: MAKING PREDICTIONS")
    print("="*70)
    
    X = df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Get predictions and probabilities
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)
    
    df['lr_prediction'] = ['select' if p == 1 else 'text' for p in predictions]
    df['lr_confidence'] = np.max(probabilities, axis=1)
    
    if 'true_kind' in df.columns:
        df['lr_correct'] = df['lr_prediction'] == df['true_kind']
        accuracy = df['lr_correct'].mean()
        print(f"\n[OK] Logistic Regression accuracy: {accuracy*100:.1f}%")
        print(f"   Correct: {df['lr_correct'].sum()}/{len(df)}")
    
    print(f"\nConfidence statistics:")
    print(f"   Mean: {df['lr_confidence'].mean():.3f}")
    print(f"   Min: {df['lr_confidence'].min():.3f}")
    print(f"   Max: {df['lr_confidence'].max():.3f}")
    
    return df


# ============================================================================
# PHASE 4: LLM FALLBACK
# ============================================================================

def find_low_confidence_cases(df, threshold=CONFIG['low_confidence_threshold']):
    """Find low-confidence cases"""
    
    low_conf = df[df['lr_confidence'] < threshold].copy()
    
    print(f"\n{'='*70}")
    print(f"PHASE 4: LLM FALLBACK ANALYSIS")
    print(f"{'='*70}")
    print(f"\nThreshold: {threshold}")
    print(f"Low-confidence cases: {len(low_conf)}/{len(df)}")
    
    if len(low_conf) > 0:
        print(f"\nLow-confidence fields:")
        for idx, field in low_conf.iterrows():
            if 'lr_correct' in field:
                correct = "[OK]" if field['lr_correct'] else "[FAIL]"
                print(f"  {correct} {field['field_id']:20} | Conf: {field['lr_confidence']:.3f} | Pred: {field['lr_prediction']}")
    else:
        print(f"\n[OK] All predictions have high confidence!")
    
    return low_conf


def create_flattened_dom(field_data):
    """Create flattened DOM representation"""
    dom = (
        f"tag:{field_data['tag']}, "
        f"type:{field_data['type']}, "
        f"opts:{int(field_data['options_count'])}, "
        f"yes_no:{int(field_data['is_yes_no_question'])}, "
        f"required:{int(field_data['is_required'])}"
    )
    return dom


def create_few_shot_examples(df, num_examples=CONFIG['num_few_shot_examples']):
    """Create few-shot examples"""
    
    if 'lr_correct' not in df.columns:
        print("[WARNING] No ground truth - skipping few-shot generation")
        return []
    
    correct_confident = df[
        (df['lr_confidence'] >= CONFIG['high_confidence_threshold']) & 
        (df['lr_correct'] == True)
    ]
    
    if len(correct_confident) == 0:
        correct_confident = df[
            (df['lr_confidence'] >= 0.90) & 
            (df['lr_correct'] == True)
        ]
    
    examples = []
    for idx, row in correct_confident.head(num_examples).iterrows():
        example = {
            "id": str(row['field_id']),
            "flattened_dom": create_flattened_dom(row.to_dict()),
            "input_kind_pred": str(row['lr_prediction']),
            "rationale": f"LR confidence: {row['lr_confidence']:.2f}"
        }
        examples.append(example)
    
    print(f"\n[OK] Created {len(examples)} few-shot examples")
    return examples


def create_phase4_prompts(low_conf_cases, few_shot_examples):
    """Create LLM prompts"""
    
    if len(low_conf_cases) == 0:
        return []
    
    print(f"\n" + "="*70)
    print(f"GENERATING PHASE 4 PROMPTS")
    print(f"="*70)
    
    prompts = []
    
    for i, (idx, field) in enumerate(low_conf_cases.iterrows(), 1):
        prompt = f"""You are a form field type classifier. Your task is to classify form fields as either "text" or "select".

## Few-Shot Examples (Ground Truth):

"""
        
        for j, ex in enumerate(few_shot_examples, 1):
            prompt += f"""Example {j}:
  ID: {ex['id']}
  DOM: {ex['flattened_dom']}
  Prediction: {ex['input_kind_pred']}
  Rationale: {ex['rationale']}

"""
        
        field_id = field['field_id']
        dom = create_flattened_dom(field.to_dict())
        ml_pred = field['lr_prediction']
        ml_conf = field['lr_confidence']
        
        prompt += f"""## Field to Classify:

ID: {field_id}
DOM: {dom}
Current ML Prediction: {ml_pred} (confidence: {ml_conf:.2f})

## Your Task:

Based on the DOM snapshot and the few-shot examples above, classify this field.

Respond ONLY with valid JSON, no additional text:

{{
  "id": "{field_id}",
  "input_kind": "text or select",
  "rationale": "1-2 sentence explanation"
}}
"""
        
        prompts.append({
            "case_number": i,
            "field_id": str(field['field_id']),
            "phase3_prediction": str(field['lr_prediction']),
            "phase3_confidence": float(field['lr_confidence']),
            "prompt": prompt
        })
    
    print(f"Created {len(prompts)} prompts")
    return prompts


def display_phase4_prompts(prompts):
    """Display prompts"""
    
    if len(prompts) == 0:
        return
    
    print(f"\n" + "="*70)
    print(f"PHASE 4: PROMPTS FOR CLAUDE")
    print(f"="*70)
    
    for prompt_data in prompts:
        print(f"\n{'='*70}")
        print(f"CASE {prompt_data['case_number']}/{len(prompts)}: {prompt_data['field_id']}")
        print(f"{'='*70}")
        print(f"Phase 3: {prompt_data['phase3_prediction']} (conf: {prompt_data['phase3_confidence']:.3f})")
        print(f"\n--- COPY FROM HERE ---\n")
        print(prompt_data['prompt'])
        print(f"\n--- PASTE INTO CLAUDE ---\n")


def load_llm_responses(filepath='llm_responses.json'):
    """Load LLM responses if they exist"""
    try:
        with open(filepath, 'r') as f:
            responses = json.load(f)
        print(f"\n[OK] Loaded {len(responses)} LLM responses")
        return responses
    except FileNotFoundError:
        return {}


def ensemble_predictions(df, llm_responses):
    """Ensemble Phase 3 + LLM"""
    
    if len(llm_responses) == 0:
        print("\n[WARNING] No LLM responses - using Phase 3 only")
        df['final_prediction'] = df['lr_prediction']
        df['final_confidence'] = df['lr_confidence']
        df['phase4_used'] = False
    else:
        print(f"\n" + "="*70)
        print(f"ENSEMBLE: COMBINING PHASE 3 + LLM")
        print(f"="*70)
        
        df['final_prediction'] = df['lr_prediction']
        df['final_confidence'] = df['lr_confidence']
        df['phase4_used'] = False
        df['llm_rationale'] = ''
        
        for field_id, llm_resp in llm_responses.items():
            mask = df['field_id'] == field_id
            if mask.sum() > 0:
                df.loc[mask, 'final_prediction'] = llm_resp['input_kind'].lower()
                df.loc[mask, 'final_confidence'] = 0.90
                df.loc[mask, 'phase4_used'] = True
                df.loc[mask, 'llm_rationale'] = llm_resp['rationale']
        
        print(f"\n[OK] Applied {len(llm_responses)} LLM responses")
    
    return df


# ============================================================================
# FINAL SUBMISSION
# ============================================================================

def calculate_final_accuracy(df):
    """Calculate final accuracy"""
    
    if 'true_kind' not in df.columns:
        print("\n[WARNING] No ground truth - skipping accuracy calculation")
        return df
    
    df['final_correct'] = df['final_prediction'] == df['true_kind']
    
    accuracy = df['final_correct'].mean()
    correct_count = df['final_correct'].sum()
    
    print(f"\n" + "="*70)
    print(f"FINAL ACCURACY")
    print(f"="*70)
    print(f"\nFinal Accuracy: {accuracy*100:.1f}% ({int(correct_count)}/{len(df)})")
    
    if 'lr_correct' in df.columns:
        phase3_acc = df['lr_correct'].mean()
        phase4_cases = df['phase4_used'].sum()
        
        print(f"Phase 3 (LR): {phase3_acc*100:.1f}%")
        if phase4_cases > 0:
            phase4_correct = df[df['phase4_used']]['final_correct'].sum()
            print(f"Phase 4 (LLM): {int(phase4_correct)}/{int(phase4_cases)} correct")
        
        if accuracy > phase3_acc:
            improvement = (accuracy - phase3_acc) * 100
            print(f"\n[OK] IMPROVEMENT: +{improvement:.1f}%")
    
    return df


def generate_final_submission(df, output_file='predictions.jsonl'):
    """Generate final submission"""
    
    print(f"\n" + "="*70)
    print(f"GENERATING FINAL SUBMISSION")
    print(f"="*70)
    
    predictions = []
    
    for idx, row in df.iterrows():
        evidence = []
        
        if row['has_options'] == 1:
            evidence.append(f"Options detected ({int(row['options_count'])})")
        else:
            evidence.append("No options detected")
        
        if row['is_yes_no_question'] == 1:
            evidence.append("Yes/no question pattern")
        
        if row['phase4_used']:
            evidence.append("LLM classified (Phase 4)")
            if row['llm_rationale']:
                evidence.append(f"LLM: {row['llm_rationale']}")
        else:
            confidence_level = row['final_confidence']
            if confidence_level >= 0.95:
                evidence.append("Very high LR confidence (>=0.95)")
            elif confidence_level >= 0.85:
                evidence.append("High LR confidence (>=0.85)")
            else:
                evidence.append(f"LR confidence: {confidence_level:.2f}")
        
        prediction = {
            "id": str(row['field_id']),
            "label": str(row['field_id']),
            "input_kind_pred": str(row['final_prediction']).lower(),
            "confidence": float(round(row['final_confidence'], 2)),
            "evidence": evidence[:3]
        }
        predictions.append(prediction)
    
    output_path = os.path.join(CONFIG['output_dir'], output_file)
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"\n[OK] Saved final submission: {output_path}")
    print(f"   Format: JSONL (one JSON per line)")
    print(f"   Fields: {len(predictions)} predictions")
    
    return predictions


def save_dataframe(df, filepath):
    """Save dataframe to CSV"""
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved: {filepath}")


def save_json(data, filepath):
    """Save data to JSON"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[OK] Saved: {filepath}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(raw_form_path, standard_form_path, output_dir='.'):
    """Complete automated pipeline"""
    
    CONFIG['output_dir'] = output_dir
    
    print("\n" + "="*70)
    print("AUTOMATED FORM TYPE CLASSIFICATION PIPELINE")
    print("="*70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    print(f"\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    raw_form = load_json_data(raw_form_path)
    standard_form = load_json_data(standard_form_path)
    
    if raw_form is None or standard_form is None:
        print("\n[ERROR] Failed to load data")
        return
    
    # Step 2: Process data
    print(f"\n" + "="*70)
    print("STEP 2: PROCESSING DATA")
    print("="*70)
    
    raw_df = process_raw_form(raw_form)
    standard_df = process_standard_form(standard_form)
    
    if len(raw_df) == 0 or len(standard_df) == 0:
        print("\n[ERROR] Failed to process data")
        return
    
    # Step 3: Rule-based baseline
    standard_df = apply_rule_engine(standard_df)
    raw_df = apply_rule_engine(raw_df)
    
    # Step 4: Train Logistic Regression
    model, scaler, feature_cols = train_logistic_regression(standard_df)
    
    # Step 5: Predict on raw form
    raw_df = predict_with_logistic_regression(raw_df, model, scaler, feature_cols)
    
    # Step 6: Identify low-confidence cases
    low_conf_cases = find_low_confidence_cases(raw_df)
    
    # Step 7: Phase 4 - LLM (if needed)
    final_df = raw_df.copy()
    
    if len(low_conf_cases) > 0:
        few_shot = create_few_shot_examples(standard_df)
        prompts = create_phase4_prompts(low_conf_cases, few_shot)
        
        # Save prompts
        prompts_path = os.path.join(output_dir, 'phase4_prompts.json')
        save_json(prompts, prompts_path)
        
        # Display prompts
        display_phase4_prompts(prompts)
        
        # Try to load responses
        llm_responses = load_llm_responses()
        final_df = ensemble_predictions(final_df, llm_responses)
    else:
        final_df['final_prediction'] = final_df['lr_prediction']
        final_df['final_confidence'] = final_df['lr_confidence']
        final_df['phase4_used'] = False
    
    # Step 8: Final accuracy
    final_df = calculate_final_accuracy(final_df)
    
    # Step 9: Generate submission
    predictions = generate_final_submission(final_df)
    
    # Step 10: Save results
    print(f"\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    results_path = os.path.join(output_dir, 'classification_results.csv')
    save_dataframe(final_df, results_path)
    
    print(f"\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nOutputs:")
    print(f"  - predictions.jsonl (final submission)")
    print(f"  - classification_results.csv (all predictions)")
    if len(low_conf_cases) > 0:
        print(f"  - phase4_prompts.json (LLM prompts)")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automated Form Type Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '-r', '--raw-form',
        default='raw_form.json',
        help='Path to raw form JSON data (default: raw_form.json)'
    )
    
    parser.add_argument(
        '-s', '--standard-form',
        default='standard_form.json',
        help='Path to standard form JSON data (default: standard_form.json)'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.85,
        help='Low-confidence threshold for Phase 4 (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    CONFIG['low_confidence_threshold'] = args.threshold
    
    if args.output_dir != '.' and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    main(args.raw_form, args.standard_form, args.output_dir)