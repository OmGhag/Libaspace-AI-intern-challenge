"""
Phase 4: LLM Zero/Few-Shot Fallback Implementation
Exact specification from challenge PDF

For low-confidence cases, use:
- Flattened DOM snapshot (compact)
- 3-5 few-shot examples (teach Claude)
- Strict JSON output
- Short rationale

Usage:
  python phase4_llm_implementation.py
"""

import pandas as pd
import json
import re
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

LOW_CONFIDENCE_THRESHOLD = 0.85
HIGH_CONFIDENCE_THRESHOLD = 0.95
NUM_EXAMPLES = 5


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_phase3_results(filepath=r'C:\Users\omgha\OneDrive\Documents\GitHub\Libaspace-AI-intern-challenge\Phase3\logistic_regression_results.csv'):
    """Load Phase 3 Logistic Regression results"""
    try:
        results = pd.read_csv(filepath)
        print(f"✅ Loaded Phase 3: {len(results)} fields")
        print(f"   Accuracy: {(results['lr_correct'].mean()*100):.1f}%")
        return results
    except FileNotFoundError:
        print(f"❌ Could not find {filepath}")
        print("   Make sure you saved Phase 3 results to CSV")
        return None


# ============================================================================
# STEP 2: CREATE FLATTENED DOM SNAPSHOT
# ============================================================================

def create_flattened_dom(field_data):
    """
    Create compact flattened DOM representation
    
    Format: tag:X, type:X, opts:X, yes_no:X, required:X
    """
    dom = (
        f"tag:{field_data['tag']}, "
        f"type:{field_data['type']}, "
        f"opts:{int(field_data['options_count'])}, "
        f"yes_no:{int(field_data['is_yes_no_question'])}, "
        f"required:{int(field_data['is_required'])}"
    )
    return dom


# ============================================================================
# STEP 3: CREATE FEW-SHOT EXAMPLES
# ============================================================================

def create_few_shot_examples(phase3_results, num_examples=NUM_EXAMPLES):
    """
    Create 3-5 high-confidence correct predictions as examples
    
    Requirements:
    - Confidence >= 0.95 (high confidence)
    - Correct prediction (matches true_kind)
    """
    
    # Get high-confidence correct predictions
    correct_confident = phase3_results[
        (phase3_results['lr_confidence'] >= HIGH_CONFIDENCE_THRESHOLD) & 
        (phase3_results['lr_correct'] == True)
    ]
    
    if len(correct_confident) == 0:
        print("⚠️  No high-confidence correct predictions found!")
        print("   Lowering threshold to 0.90...")
        correct_confident = phase3_results[
            (phase3_results['lr_confidence'] >= 0.90) & 
            (phase3_results['lr_correct'] == True)
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
    
    print(f"\n✅ Created {len(examples)} few-shot examples")
    return examples


# ============================================================================
# STEP 4: CREATE LLM PROMPT WITH FLATTENED DOM + FEW-SHOT
# ============================================================================

def create_phase4_prompt(field_data, few_shot_examples):
    """
    Create LLM prompt with:
    - Flattened DOM snapshot (compact)
    - 3-5 few-shot examples
    - Request for JSON output
    - Short rationale
    """
    
    prompt = """You are a form field type classifier. Your task is to classify form fields as either "text" or "select".

## Few-Shot Examples (Ground Truth):

"""
    
    # Add each example with flattened DOM
    for i, ex in enumerate(few_shot_examples, 1):
        prompt += f"""Example {i}:
  ID: {ex['id']}
  DOM: {ex['flattened_dom']}
  Prediction: {ex['input_kind_pred']}
  Rationale: {ex['rationale']}

"""
    
    # Add field to classify
    field_id = field_data['field_id']
    dom = create_flattened_dom(field_data)
    ml_pred = field_data['lr_prediction']
    ml_conf = field_data['lr_confidence']
    
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
    
    return prompt


# ============================================================================
# STEP 5: IDENTIFY LOW-CONFIDENCE CASES
# ============================================================================

def find_low_confidence_cases(phase3_results, threshold=LOW_CONFIDENCE_THRESHOLD):
    """Find cases where Phase 3 is uncertain (confidence < threshold)"""
    
    low_conf = phase3_results[
        phase3_results['lr_confidence'] < threshold
    ].copy()
    
    print(f"\n{'='*70}")
    print(f"LOW-CONFIDENCE CASE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nThreshold: {threshold}")
    print(f"Low-confidence cases: {len(low_conf)}/{len(phase3_results)}")
    
    if len(low_conf) > 0:
        print(f"\nFields needing LLM help:")
        for idx, field in low_conf.iterrows():
            correct = "✅" if field['lr_correct'] else "❌"
            print(f"  {correct} {field['field_id']:20} | Conf: {field['lr_confidence']:.3f} | Pred: {field['lr_prediction']}")
    else:
        print(f"\n✅ All predictions have high confidence!")
        print(f"   No Phase 4 needed. Use Phase 3 directly.")
    
    return low_conf


# ============================================================================
# STEP 6: SHOW PROMPTS FOR MANUAL CLASSIFICATION
# ============================================================================

def show_prompts_for_classification(low_conf_cases, few_shot_examples):
    """
    Display prompts for each low-confidence case
    User will manually copy-paste into claude.ai and get JSON responses
    """
    
    if len(low_conf_cases) == 0:
        print("\n✅ No low-confidence cases - skipping Phase 4")
        return None
    
    prompts = []
    
    print(f"\n{'='*70}")
    print(f"PHASE 4: PROMPTS FOR CLAUDE")
    print(f"{'='*70}")
    print(f"\n📋 Instructions:")
    print(f"1. Copy each prompt below (one at a time)")
    print(f"2. Go to https://claude.ai")
    print(f"3. Paste prompt into a new conversation")
    print(f"4. Copy Claude's JSON response")
    print(f"5. Paste response into llm_responses.json file (see below)")
    print(f"\nTotal prompts to process: {len(low_conf_cases)}")
    
    for i, (idx, field) in enumerate(low_conf_cases.iterrows(), 1):
        prompt = create_phase4_prompt(field.to_dict(), few_shot_examples)
        
        prompts.append({
            "case_number": i,
            "field_id": str(field['field_id']),
            "phase3_prediction": str(field['lr_prediction']),
            "phase3_confidence": float(field['lr_confidence']),
            "prompt": prompt
        })
        
        print(f"\n{'='*70}")
        print(f"CASE {i}/{len(low_conf_cases)}: {field['field_id']}")
        print(f"{'='*70}")
        print(f"Phase 3 Prediction: {field['lr_prediction']} (confidence: {field['lr_confidence']:.3f})")
        print(f"\n--- COPY FROM HERE ---\n")
        print(prompt)
        print(f"\n--- PASTE INTO CLAUDE ---\n")
        print(f"Then save Claude's JSON response to llm_responses.json")
    
    return prompts


# ============================================================================
# STEP 7: PARSE LLM RESPONSES
# ============================================================================

def parse_llm_response(response_text):
    """
    Extract JSON from Claude's response
    Claude might include extra text, so we extract just the JSON
    """
    try:
        # Try direct JSON parse (Claude returned just JSON)
        result = json.loads(response_text.strip())
        return result
    except:
        # Try to extract JSON from response (Claude included explanation)
        json_match = re.search(r'\{[^{}]*"id"[^{}]*"input_kind"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Try more aggressive extraction
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        return None


def load_llm_responses(filepath='llm_responses.json'):
    """Load LLM responses from JSON file"""
    try:
        with open(filepath, 'r') as f:
            responses = json.load(f)
        print(f"✅ Loaded {len(responses)} LLM responses from {filepath}")
        return responses
    except FileNotFoundError:
        print(f"\n⚠️  Could not find {filepath}")
        print(f"   Create this file with Claude's JSON responses")
        print(f"   Format: {{'field_id': {{'input_kind': 'text', 'rationale': '...'}}, ...}}")
        return {}


# ============================================================================
# STEP 8: ENSEMBLE - COMBINE PHASE 3 + LLM
# ============================================================================

def ensemble_predictions(phase3_results, llm_responses):
    """
    Combine Phase 3 and LLM predictions
    
    Logic:
    - If Phase 3 confident (>= 0.85): use Phase 3
    - If Phase 3 uncertain (< 0.85) and LLM provided: use LLM
    - Otherwise: use Phase 3 as fallback
    """
    
    final_results = phase3_results.copy()
    final_results['final_prediction'] = final_results['lr_prediction']
    final_results['final_confidence'] = final_results['lr_confidence']
    final_results['phase4_used'] = False
    final_results['llm_rationale'] = ''
    
    # Apply LLM responses
    for field_id, llm_resp in llm_responses.items():
        mask = final_results['field_id'] == field_id
        
        if mask.sum() > 0:
            # Update with LLM prediction
            final_results.loc[mask, 'final_prediction'] = llm_resp['input_kind'].lower()
            final_results.loc[mask, 'final_confidence'] = 0.90  # Trust LLM over uncertain ML
            final_results.loc[mask, 'phase4_used'] = True
            final_results.loc[mask, 'llm_rationale'] = llm_resp['rationale']
    
    return final_results


# ============================================================================
# STEP 9: CALCULATE FINAL ACCURACY
# ============================================================================

def calculate_accuracy(final_results):
    """Calculate final accuracy with Phase 3 + Phase 4"""
    
    final_results['final_correct'] = (
        final_results['final_prediction'] == final_results['true_kind']
    )
    
    accuracy = final_results['final_correct'].mean()
    correct_count = final_results['final_correct'].sum()
    
    # Statistics
    phase3_acc = (final_results['lr_correct']).mean()
    phase4_cases = final_results['phase4_used'].sum()
    phase4_correct = final_results[final_results['phase4_used']]['final_correct'].sum()
    
    print(f"\n{'='*70}")
    print(f"FINAL ACCURACY SUMMARY")
    print(f"{'='*70}")
    print(f"\nPhase 3 (Logistic Regression): {phase3_acc*100:.1f}% ({final_results['lr_correct'].sum()}/23)")
    print(f"Phase 4 (LLM) cases processed: {phase4_cases}")
    if phase4_cases > 0:
        print(f"Phase 4 correct: {int(phase4_correct)}/{int(phase4_cases)}")
    print(f"\nFinal Accuracy: {accuracy*100:.1f}% ({int(correct_count)}/23)")
    
    if accuracy > phase3_acc:
        improvement = (accuracy - phase3_acc) * 100
        print(f"✅ IMPROVEMENT: +{improvement:.1f}%")
    elif accuracy == phase3_acc:
        print(f"→ No change from Phase 3")
    else:
        degradation = (phase3_acc - accuracy) * 100
        print(f"⚠️  Degradation: -{degradation:.1f}%")
    
    return final_results


# ============================================================================
# STEP 10: GENERATE FINAL SUBMISSION
# ============================================================================

def generate_final_submission(final_results, output_file='predictions.jsonl'):
    """
    Generate final submission in correct format
    One JSON object per line (JSONL format)
    
    Fields:
    - id: field identifier
    - label: field label (using id for now)
    - input_kind_pred: prediction (text or select)
    - confidence: confidence score
    - evidence: list of reasons
    """
    
    predictions = []
    
    for idx, row in final_results.iterrows():
        # Generate evidence
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
                evidence.append(f"LLM rationale: {row['llm_rationale']}")
        else:
            confidence_level = row['final_confidence']
            if confidence_level >= 0.95:
                evidence.append("Very high LR confidence (≥0.95)")
            elif confidence_level >= 0.85:
                evidence.append("High LR confidence (≥0.85)")
            else:
                evidence.append(f"LR confidence: {confidence_level:.2f}")
        
        prediction = {
            "id": str(row['field_id']),
            "label": str(row['field_id']),
            "input_kind_pred": str(row['final_prediction']).lower(),
            "confidence": float(round(row['final_confidence'], 2)),
            "evidence": evidence
        }
        predictions.append(prediction)
    
    # Save as JSONL
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    print(f"\n✅ Saved final submission: {output_file}")
    print(f"   Format: JSONL (one JSON per line)")
    print(f"   Fields: 23 predictions")
    
    return predictions


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete Phase 4 workflow"""
    
    print("="*70)
    print("PHASE 4: LLM ZERO/FEW-SHOT FALLBACK IMPLEMENTATION")
    print("="*70)
    print("\nChallenge Specification:")
    print("- Use flattened DOM snapshot")
    print("- Include 3-5 few-shot examples")
    print("- Request strict JSON output")
    print("- Include short rationale")
    
    # Step 1: Load Phase 3
    phase3_results = load_phase3_results()
    if phase3_results is None:
        return
    
    # Step 2: Find low-confidence cases
    low_conf_cases = find_low_confidence_cases(phase3_results)
    
    # Check if Phase 4 needed
    if len(low_conf_cases) == 0:
        print(f"\n✅ All predictions confident - no Phase 4 needed")
        final_results = phase3_results.copy()
        final_results['final_prediction'] = final_results['lr_prediction']
        final_results['final_confidence'] = final_results['lr_confidence']
        final_results['phase4_used'] = False
    else:
        # Step 3: Create few-shot examples
        few_shot = create_few_shot_examples(phase3_results)
        
        # Step 4: Show prompts for manual classification
        prompts = show_prompts_for_classification(low_conf_cases, few_shot)
        
        # Save prompts for reference
        if prompts:
            with open('phase4_prompts.json', 'w') as f:
                json.dump(prompts, f, indent=2)
            print(f"\n✅ Saved prompts to: phase4_prompts.json")
        
        # Step 5: Wait for user to provide LLM responses
        print(f"\n" + "="*70)
        print(f"NEXT STEPS:")
        print(f"="*70)
        print(f"\n1. Copy each prompt above and paste into https://claude.ai")
        print(f"2. Copy Claude's JSON response")
        print(f"3. Create llm_responses.json with format:")
        print(f"""
   {{
     "field_id_1": {{"input_kind": "text", "rationale": "explanation"}},
     "field_id_2": {{"input_kind": "select", "rationale": "explanation"}},
     ...
   }}
""")
        print(f"4. Run this script again: python phase4_llm_implementation.py")
        
        # Try to load responses if they exist
        llm_responses = load_llm_responses()
        
        if len(llm_responses) > 0:
            # Step 6: Ensemble predictions
            final_results = ensemble_predictions(phase3_results, llm_responses)
        else:
            print(f"\n⏳ Waiting for LLM responses...")
            print(f"   Once you have llm_responses.json, run this script again")
            return
    
    # Step 7: Calculate accuracy
    final_results = calculate_accuracy(final_results)
    
    # Step 8: Generate final submission
    predictions = generate_final_submission(final_results)
    
    print(f"\n" + "="*70)
    print(f"✅ PHASE 4 COMPLETE!")
    print(f"="*70)
    print(f"\nSubmission ready: predictions.jsonl")
    print(f"Ready to submit to challenge!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()