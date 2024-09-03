import json
import logging
from typing import List, Tuple, Dict, Any

import pandas as pd
import torch
from overrides import override
from torch.utils.data.dataset import ConcatDataset, Dataset

from ihemomil.data.mil_tsc_dataset import MILTSCDataset

logger = logging.getLogger("IHEMOMIL.MorpperturPPGDataset")

class MorpperturPPGDataset(MILTSCDataset):
    
    def __init__(self, split: str, name: str = "VitalDB", apply_transform: bool = True):
        super().__init__(name, split, apply_transform=apply_transform)
        # Load dataset metadata
        metadata_path = "data/MorpPerturPPG/{:s}_{:s}_metadata.json".format(self.dataset_name, split.upper())
        with open(metadata_path, "r") as f:
            self._metadata: List[Dict[str, Any]] = json.load(f)
        self.ts = pd.read_csv("data/MorpPerturPPG/Raw_data/{:s}_{:s}_Raw.csv".format(self.dataset_name, split.upper()))
    
    def get_time_series_collection_and_targets(self, split: str) -> Tuple[List[torch.Tensor], torch.Tensor]:
        df = pd.read_csv("data/MorpPerturPPG/{:s}_{:s}.csv".format(self.dataset_name, split.upper()))
        ts_collection = []
        targets = []
        for row_idx in range(len(df)):
            ts_pd = df.iloc[row_idx, 1:]
            target = df.iloc[row_idx, 0]
            ts_tensor = torch.as_tensor(ts_pd.to_numpy(), dtype=torch.float)
            ts_tensor = ts_tensor.unsqueeze(1)
            ts_collection.append(ts_tensor)
            targets.append(target)
        targets_tensor = torch.as_tensor(targets, dtype=torch.int)
        return ts_collection, targets_tensor
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        return self._metadata
    
    def get_signature_locations(self, idx: int) -> List:
        return self._metadata[idx]["signature_locations"]
    
    def get_original_time_series(self, idx: int) -> torch.Tensor:
        return torch.as_tensor(self.ts.iloc[idx, 1:].to_numpy(), dtype=torch.float)
 

    @override
    def __getitem__(self, idx: int) -> Dict:
        # Get bag and target
        bag = self.get_bag(idx)
        if self.apply_transform:
            bag = self.apply_bag_transform(bag)
        target = self.targets[idx]
        # Convert signature locations to instance targets
        if target == 0:
            instance_targets = torch.ones(len(bag))
        else:
            signature_locs = self.get_signature_locations(idx)
            instance_targets = torch.zeros(len(bag))
            for signature_loc in signature_locs:
                sig_start, sig_end = signature_loc
                if sig_start == sig_end:
                    instance_targets[sig_start] = 1
                else:
                    instance_targets[sig_start:sig_end] = 1
        return {"bag": bag, 
                "target": target, 
                "instance_targets": instance_targets,
        }