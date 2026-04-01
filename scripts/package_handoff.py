import os
import zipfile

def create_model_package():
    print("Preparing ML Model Package for API Integration...")
    
    # Files and folders to include
    targets = [
        "requirements.txt",
        ".env.example",
        "ml_service",
        "data/structured",
        "model/active",
        "model/registry.json",
        "docs/phase7", 
        "data/test_requests.json",
        "tests"
    ]
    
    output_filename = "ML_Costing_Release_v1.0.zip"
    
    # Remove existing zip if any
    if os.path.exists(output_filename):
        os.remove(output_filename)
        
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for target in targets:
            abs_target = os.path.abspath(target)
            if not os.path.exists(abs_target):
                print(f"File or directory not found: {target}")
                continue
                
            if os.path.isfile(abs_target):
                zipf.write(abs_target, target)
                print(f"Added {target}")
            elif os.path.isdir(abs_target):
                for root, _, files in os.walk(abs_target):
                    # Filter out caches and pycache
                    if '__pycache__' in root or '.pytest_cache' in root:
                        continue
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, os.getcwd())
                        zipf.write(file_path, rel_path)
    
    print(f"\n✅ Package successfully created: {output_filename}")
    
if __name__ == "__main__":
    create_model_package()
