import torch
from pathlib import Path
from typing import Dict
from cprint import *
from datamodule.dataset.base import DATASET, DatasetType, MKPointCloudStateBase
from datamodule.dataset.transforms import make_default_transform

@DATASET.register()
class MKPointCloudTrajectoryDataset(MKPointCloudStateBase):
    """ This dataset is used exclusively for M2Diffuser training and evaluation, and MPiNets and evaluation. 
    Each element in the dataset represents a trajectory start and scene. There is no supervision because this 
    is used to produce an entire rollout and check for success. When doing validation, we care more about 
    success than we care about matching the expert's behavior (which is a key difference from training).
    """

    def __init__(
        self,
        cfg: dict, 
        data_dir: Path, 
        dataset_type: DatasetType, 
        **kwargs: Dict,
    ):
        """ The function initializes the dataset instance with configuration, data directory, and dataset type, 
        and sets up the default data transformation.

        Args:
            cfg [dict]: Configuration dictionary for the dataset.
            data_dir [Path]: The path to the root of the data directory.
            dataset_type [DatasetType]: The type of dataset (e.g., train, val, test).
            **kwargs [Dict]: Additional keyword arguments for dataset customization.

        Return:
            None: This is an initializer and does not return a value.
        """
        super().__init__(cfg, data_dir, dataset_type, **kwargs)
        self.transform = make_default_transform(cfg, dataset_type)

    def __len__(self):
        """ Returns the total number of expert motions in the dataset.
        """
        return self.num_trajectories
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """ This function retrieves a data sample at a specific index, corresponding to a start configuration 
        (for MPiNets training) and complete trajectory (for M2Diffuser training and evaluation) in the dataset. 

        Args:
            idx [int]: The index of the trajectory to retrieve.

        Return:
            Dict[str, torch.Tensor]: A dictionary containing the trajectory sample.
        """
        trajectory_idx, timestep = idx, 0
        item = self.get_inputs(trajectory_idx, timestep)
        if self.transform is not None:
            item = self.transform(item)
        return item