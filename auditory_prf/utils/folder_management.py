# utils/folder_management.py
"""
Folder creation and management utilities for experiment workflows.

Contains:
    - FolderCreator: Low-level filesystem operations
    - FolderManager: High-level experiment folder orchestration
"""

import os
from .metadata_saver import MetadataSaver
from .timestamp_utils import generate_timestamp, TimestampFormats
from .model_builders import model_builders


class FolderCreator(object):
    """
    Create directories on the filesystem.
    """
    def __init__(self, base_dir: str):
        """
        Args:
            base_dir (str): Base directory where folders will be created.
        """
        self.base_dir = base_dir

    def create_folder(self, folder_name: str) -> str:
        """
        Docstring for create_folder

        :param folder_name: Description
        """
        folder_path = os.path.join(self.base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def create_subfolder(self, parent_folder: str, subfolder_name: str) -> str:
        """
        Docstring for create_subfolder

        :param self: Description
        :param parent_folder: Description
        :type parent_folder: str
        :param subfolder_name: Description
        :type subfolder_name: str
        :return: Description
        :rtype: str
        """
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        return subfolder_path


class FolderManager(object):
    """
    Docstring for FolderManager

    Args:
        base_dir: Base directory where folders will be created
        name_builder: Either a string (registered in model_builders) or a
        callable that takes (params, timestamp) and returns a folder name
    """

    def __init__(self, base_dir: str, name_builder=None):

        self.params = {}
        self.name_builder = name_builder
        self.results_folder = None
        self.timestamp_format = TimestampFormats.COMPACT

        # Composition: inject dependencies
        self.folder_creator = FolderCreator(base_dir)
        self.metadata_saver = MetadataSaver()


    def with_params(self, **kwargs):
        self.params.update(kwargs)
        return self

    def with_timestamp_format(self, format_string):
        self.timestamp_format = format_string
        return self

    def create_folder(self, folder_name=None, save_json=True, save_text=True):
        """
        Orchestrate the complete folder creation workflow with experiment
        metadata.

        :param self: Description
        :param folder_name: Description
        :param save_json: Description
        :param save_text: Description
        """

        # 1. Generate timestamp
        timestamp = generate_timestamp(self.timestamp_format)

        # 2. Build folder name
        if folder_name is None:
            if isinstance(self.name_builder, str):
                # Default: get builder from registry
                builder_func = model_builders.get(self.name_builder)
                folder_name = builder_func(self.params, timestamp)
            elif self.name_builder is None:
                folder_name = f"results_{timestamp}"
            else:
                # Use custom callable
                folder_name = self.name_builder(self.params, timestamp)
        # 3. Create folder
        self.results_folder = self.folder_creator.create_folder(folder_name)

        # 4. Save metadata files
        if save_json:
            self.metadata_saver.save_json(self.results_folder, self.params)
        if save_text:
            self.metadata_saver.save_text(self.results_folder, self.params)

        return self.results_folder

    def create_subfolder(self, subfolder_name: str):
        """
        Create a subfolder within the results folder.

        :param self: Description
        :param subfolder_name: Description
        :type subfolder_name: str
        """

        if not self.results_folder:
            raise RuntimeError("Results folder not created yet.",
                               "Call create_folder() first.")
        return self.folder_creator.create_subfolder(self.results_folder,
                                                    subfolder_name)

    def get_results_folder(self):
        """
        Docstring for get_results_folder

        :param self: Description
        """
        return self.results_folder


# Backward compatibility alias
ExperimentFolderManager = FolderManager


# Usage examples:
#
# Using model name from registry:
# manager = (FolderManager("./results", "bez2018")
#           .with_params(num_runs=10, num_cf=20, min_cf=125,
#                       num_ANF=(4,4,4))
#           .create_folder())

# using cochlea model:
# manager = (FolderManager("./results", "cochlea_zilany2014")
#           .with_params(num_runs=5, num_cf=20, min_cf=125, max_cf=2500)
#           .create_folder())

# using WSR model:
# manager = (FolderManager("./results", "wsr_model")
#           .with_params(num_channels=128, frame_length=16,
#                       time_constant=8, factor-2, shift=4)
#                       .create_folder())

# Custom timestamp format:
# manager = (FolderManager("./results", "bez2018")
#           .with_timestamp_format("%Y-%m-%d_%H-%M-%S")
#           .with_params(num_runs=10, num_cf=20, min_cf=125,
#                       num_ANF=(4, 4, 4))
#           .create_folder())

# Custom name builder function:
# def custom_namer(params, timestamp):
#     return f"analysis_{params['name']}_{timestamp}"
# manager = FolderManager("./results", custom_namer).with_params(name="test")
#                                                      .create_folder()

# Simple folder with explicit name:
# manager = FolderManager("./results").create_folder("my_analysis_20250101")

# Simple folder with just timestamp:
# manager = FolderManager("./results").create_folder()

# Create subfolders:

# manager.create_subfolder("plots")
# manager.create_subfolder("raw_data")
