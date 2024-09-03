import re
import json
import os
import os.path
import numpy as np
import pandas as pd
from _ctypes import PyObj_FromPtr  

REALFLUDPPG_CLZ_NAMES = [
    "normal",
    "hypertension",
    "hypotension",
]


usage_mode = ["train", "test"]


def load_and_refactor_data():
    """
    load_and_refactor_data
    """
    parent_path = "/home/danielwang_echo/Workspace/IHemoMIL/data/PulseDBBPF"
    data_file_path = os.listdir(parent_path)
    subject_name = list(set(data_file_path[i].split('_')[0] for i in range(len(data_file_path))))

    save_dir = "/home/danielwang_echo/Workspace/IHemoMIL/data/PulseDBBPF"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for split in usage_mode:
        print("Processing {:s} data".format(split))
        
        subject_data_file_name = subject_name[0] + "_" + split.upper() + ".csv"
        subject_data_file_path = os.path.join(parent_path, subject_data_file_name)
        subject_data = pd.read_csv(subject_data_file_path)
        metadata = []
        data_item_idx = 0
        for item in range(len(subject_data)):
            current_data_item = subject_data.iloc[item, :].to_numpy()
            metadata.append(
                {
                    "idx": data_item_idx,
                    "clz": int(current_data_item[0]),
                    "signature": REALFLUDPPG_CLZ_NAMES[int(current_data_item[0])],
                    "signature_locations": [current_data_item[-2:].astype(np.int32).tolist()],
                }
            )
            data_item_idx += 1
            
        save_metadata_name = "{:s}_{:s}_metadata.json".format(subject_name[0], split.upper())
        with open(os.path.join(save_dir, save_metadata_name), "w") as f:
                json.dump(metadata, f, indent=2)


load_and_refactor_data()
        
