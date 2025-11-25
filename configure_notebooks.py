#!/usr/bin/env python3
"""
Script to configure multiple Jupyter notebooks with default Python kernel
Run this to avoid kernel selection prompts for commonly used notebooks
"""

import os
import json
import glob

def configure_notebook_kernel(notebook_path):
    """Add kernel specification to notebook metadata"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Update kernel specification
        if 'metadata' not in notebook:
            notebook['metadata'] = {}
        
        notebook['metadata']['kernelspec'] = {
            "display_name": "Python 3",
            "language": "python", 
            "name": "python3"
        }
        
        notebook['metadata']['language_info'] = {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.13.7"
        }
        
        # Write back to file
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
            
        print(f"✅ Configured: {os.path.basename(notebook_path)}")
        return True
        
    except Exception as e:
        print(f"❌ Error configuring {notebook_path}: {str(e)}")
        return False

def main():
    """Configure all notebooks in common ML directories"""
    base_path = "d:/Ashvad/AIML/PoCs/practice/ML"
    
    # Common directories with notebooks
    directories = [
        "1_linear_reg/Exercise",
        "2_linear_reg_multivariate/Exercise", 
        "3_gradient_descent",
        "4_save_model",
        "5_dummy_variables",
        "6_onehotencoding",
        "7_training_testing_data_split"
    ]
    
    total_configured = 0
    
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        if os.path.exists(full_path):
            notebooks = glob.glob(os.path.join(full_path, "*.ipynb"))
            for notebook in notebooks:
                if configure_notebook_kernel(notebook):
                    total_configured += 1
        else:
            print(f"📁 Directory not found: {full_path}")
    
    print(f"\\n🎉 Total notebooks configured: {total_configured}")
    print("\\n💡 Tip: Restart VS Code to ensure all settings take effect!")

if __name__ == "__main__":
    main()