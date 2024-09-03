
import json
import logging
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from overrides import override
from torch.utils.data.dataset import ConcatDataset, Dataset

from ihemomil.data.mil_tsc_dataset import MILTSCDataset

logger = logging.getLogger("IHEMOMIL.PERFormAFDATASET")

MIMIC_AF_CLZ_NAMES = [
    "normal",
    "afib",
]

class PERFormAFDATASET(MILTSCDataset):
    
    def __init__(self, split: str, name: str = "VitalDB", apply_transform: bool = True):
        super().__init__(name, split, apply_transform=apply_transform)
    
    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        npz = np.load("data/MIMICPERFormAF/{:s}_{:s}.npy".format(self.dataset_name, split.upper()))
        npz = npz[~np.isnan(npz).any(axis=1)]
        cols = npz.shape[1]
        npz = npz[:, [cols-1] + list(range(cols-1))]
        df = pd.DataFrame(
            columns = ["Class"] + ["t_{:d}".format(f) for f in range(cols-1)],
            data = npz 
            )
        
        ts_collection = []
        targets = []
        for row_idx in range(len(df)):
            ts_pd = df.iloc[row_idx, 1:]
            target = df.iloc[row_idx, 0].astype(int)
            ts_tensor = torch.as_tensor(ts_pd.to_numpy(), dtype=torch.float)
            ts_tensor = ts_tensor.unsqueeze(1)
            ts_collection.append(ts_tensor)
            targets.append(target)
        targets_tensor = torch.as_tensor(targets, dtype=torch.int)
        return ts_collection, targets_tensor
    
    @override
    def __getitem__(self, idx: int) -> Dict:
        # Get bag and target
        bag = self.get_bag(idx)
        if self.apply_transform:
            bag = self.apply_bag_transform(bag)
        target = self.targets[idx]
        instance_targets = torch.ones(len(bag))

        return {"bag": bag, 
                "target": target, 
                "instance_targets": instance_targets,
        }