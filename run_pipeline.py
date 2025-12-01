"""
Wrapper script for automated pipeline
Handles data.json format with raw_form and standard_form keys
"""

import json
import sys
import subprocess
import os

def extract_and_run(data_file='data.json'):
    """
    Extract raw_form and standard_form from data.json
    Create temporary JSON files
    Run automated pipeline
    """
    
    print("\n" + "="*70)
    print("PIPELINE WRAPPER - EXTRACTING DATA")
    print("="*70)
    
    # Load combined data
    print(f"\n[INFO] Loading: {data_file}")
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {data_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON: {e}")
        return False
    
    # Check structure
    if not isinstance(data, dict):
        print(f"[ERROR] Expected dict, got {type(data).__name__}")
        return False
    
    if 'raw_form' not in data or 'standard_form' not in data:
        print(f"[ERROR] Missing 'raw_form' or 'standard_form' keys")
        print(f"[INFO] Found keys: {list(data.keys())}")
        return False
    
    raw_form = data['raw_form']
    standard_form = data['standard_form']
    
    print(f"[OK] Loaded data")
    print(f"   raw_form: {len(raw_form)} fields")
    print(f"   standard_form: {len(standard_form)} fields")
    
    # Extract to temporary files
    print(f"\n[INFO] Extracting to temporary files")
    
    raw_form_file = 'temp_raw_form.json'
    standard_form_file = 'temp_standard_form.json'
    
    with open(raw_form_file, 'w') as f:
        json.dump(raw_form, f)
    print(f"[OK] Created: {raw_form_file}")
    
    with open(standard_form_file, 'w') as f:
        json.dump(standard_form, f)
    print(f"[OK] Created: {standard_form_file}")
    
    # Run pipeline
    print(f"\n" + "="*70)
    print("RUNNING AUTOMATED PIPELINE")
    print("="*70)
    
    cmd = [
        'python', 'automated_pipeline.py',
        '-r', raw_form_file,
        '-s', standard_form_file,
        '-o', './results'
    ]
    
    print(f"\n[INFO] Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd)
    
    # Cleanup
    print(f"\n[INFO] Cleaning up temporary files")
    try:
        os.remove(raw_form_file)
        os.remove(standard_form_file)
        print(f"[OK] Removed temporary files")
    except:
        pass
    
    if result.returncode == 0:
        print(f"\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"\n[OK] Results saved to ./results/")
        print(f"   - predictions.jsonl (final submission)")
        print(f"   - classification_results.csv (detailed results)")
        return True
    else:
        print(f"\n[ERROR] Pipeline failed with return code {result.returncode}")
        return False

if __name__ == "__main__":
    # Check if data file provided
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'data.json'
    
    print("\n" + "="*70)
    print("FORM CLASSIFICATION PIPELINE WRAPPER")
    print("="*70)
    print(f"\n[INFO] Data file: {data_file}")
    print(f"[INFO] Expected format:")
    print(f"  {{")
    print(f"    'raw_form': [...],")
    print(f"    'standard_form': [...]")
    print(f"  }}")
    
    success = extract_and_run(data_file)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)