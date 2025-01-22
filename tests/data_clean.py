from modules.feature_ext import combine_data, compute_psd

def test_from_txt(test_file: str):
    with open(test_file, "r") as f:
        lines = f.readlines()
    
    current_test = {}
    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue  # Skip comments and empty lines
        
        if ":" in line:
            key, value = line.split(":", 1)
            current_test[key.strip()] = value.strip()
        
        if "expected_behavior" in current_test:
            try:
                print(f"Running test with dataset_dir: {current_test['dataset_dir']}")
                raw = combine_data(current_test['dataset_dir'])
                compute_psd(raw)
                print("Test passed!")
            except Exception as e:
                if current_test["expected_behavior"] in str(e):
                    print("Test passed!")
                else:
                    print(f"Test failed: {e}")
            finally:
                current_test = {}

# Run tests
test_from_txt("test_cases.txt")
