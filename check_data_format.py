"""
Diagnostic tool to check your JSON data format
Shows you what format your data is in
"""

import json
import sys

def check_json_format(filepath):
    """Check JSON file format"""
    
    print(f"\n{'='*70}")
    print(f"CHECKING: {filepath}")
    print(f"{'='*70}")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"\n✅ Valid JSON")
        print(f"\nTop-level type: {type(data).__name__}")
        print(f"Length: {len(data)}")
        
        if isinstance(data, list):
            print(f"\n📋 Format: LIST of items")
            print(f"Number of items: {len(data)}")
            
            if len(data) > 0:
                first = data[0]
                print(f"First item type: {type(first).__name__}")
                
                if isinstance(first, dict):
                    print(f"First item keys: {list(first.keys())}")
                    print(f"\nFirst item (preview):")
                    print(json.dumps(first, indent=2)[:500])
                elif isinstance(first, str):
                    print(f"First item (string):")
                    print(first[:200])
                    # Try to parse it
                    try:
                        parsed = json.loads(first)
                        print(f"\n✅ String is valid JSON!")
                        print(f"Parsed type: {type(parsed).__name__}")
                        if isinstance(parsed, dict):
                            print(f"Keys: {list(parsed.keys())}")
                    except:
                        print(f"❌ String is not valid JSON")
        
        elif isinstance(data, dict):
            print(f"\n📋 Format: DICT (object)")
            print(f"Top-level keys: {list(data.keys())[:5]}")
            
            # Check first value
            first_key = list(data.keys())[0] if data else None
            if first_key:
                first_val = data[first_key]
                print(f"\nFirst value type: {type(first_val).__name__}")
                print(f"First value (preview):")
                if isinstance(first_val, dict):
                    print(json.dumps(first_val, indent=2)[:500])
                else:
                    print(str(first_val)[:200])
        
        # Count field-like objects
        print(f"\n{'='*70}")
        print("DATA SUMMARY")
        print(f"{'='*70}")
        
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            sample = data[0]
            print(f"\nSample field object:")
            for key, val in list(sample.items())[:10]:
                print(f"  {key}: {type(val).__name__} = {str(val)[:50]}")
        
        print(f"\n✅ Data format looks valid!")
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Invalid JSON: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n❌ File not found: {filepath}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_data_format.py <json_file>")
        print("\nExample:")
        print("  python check_data_format.py data.json")
        print("  python check_data_format.py raw_form.json")
        sys.exit(1)
    
    filepath = sys.argv[1]
    check_json_format(filepath)