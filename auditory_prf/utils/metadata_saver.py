"""
Save metadata about experiment parameters to several file formats.
"""

import json
import os
import numpy as np

class MetadataSaver:
    
    @staticmethod
    def _convert_to_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: MetadataSaver._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [MetadataSaver._convert_to_serializable(item) for item in obj]
        return obj
    
    def save_json(self, folder_path, data, filename="params.json"):
        """
        Save metadata dictionary to a JSON file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the JSON file (default: "params.json")
        """
        
        filepath = os.path.join(folder_path, filename)
        # Convert numpy types to Python native types
        serializable_data = self._convert_to_serializable(data)
        with open(filepath, "w") as f:
            json.dump(serializable_data, f, indent=4)
        print(f"[INFO] Saved metadata to {filepath}")
        return filepath
        
    def save_text(self, folder_path, data, filename="stimulus_params.txt"):
        """
        Save metadata dictionary to a text file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the text file (default: "stimulus_params.txt")
        """
        
        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as f:
            for key, value in data.items():
                f.write(f"{key}={value}\n")
        print(f"[INFO] Saved stimulus params to {filepath}")
        return filepath 
    
    def save_yaml(self, folder_path, data, filename="params.yaml"):
        """
        Save metadata dictionary to a YAML file.

        Parameters:
        - folder_path: Directory where the file will be saved
        - data: Dictionary containing metadata
        - filename: Name of the YAML file (default: "params.yaml")
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save YAML files. Install it via 'pip install pyyaml'.")

        filepath = os.path.join(folder_path, filename)
        with open(filepath, "w") as f:
            yaml.dump(data, f)
        print(f"[INFO] Saved metadata to {filepath}")
        return filepath