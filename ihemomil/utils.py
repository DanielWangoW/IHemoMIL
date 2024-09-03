
import os
import sys
import logging
from tqdm import tqdm
from functools import partial

import torch
from torch import nn
from torch.nn.functional import normalize

from ihemomil.model import backbone, pooling
from ihemomil.data.web_traffic_generation import WEBTRAFFIC_CLZ_NAMES
from ihemomil.data.morppertur_ppg_generation import MORPPERTURPPG_CLZ_NAMES
from ihemomil.data.mimic_perform_af_dataset import PERFormAFDATASET, MIMIC_AF_CLZ_NAMES
from ihemomil.data.dalia_ppg_dataset import DaLiADATASET, ACTIVITY_CLZ_NAMES
from ihemomil.data.pulsedb_ppg_dataset import PulseDBPPGDataset, PulseDBBPF_CLZ_NAMES
from ihemomil.data.simhf3k_ppg_dataset import SimHF3KDataset, PERTUR_CLZ_NAMES

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from texttable import Texttable
# tqdm wrapper to write to stdout rather than stderr
custom_tqdm = partial(tqdm, file=sys.stdout)

logger = logging.getLogger("IHEMOMIL.Utils")

# Backbone Zoo
BACKBONE_ZOO = {
    "fcn": backbone.FCNFeatureExtractor,
    "inceptiontime": backbone.InceptionTimeFeatureExtractor,
    "resnet": backbone.ResNetFeatureExtractor,
    "mlp": backbone.MLPFeatureExtractor,
    "transformer": backbone.TransformerFeatureExtractor
}

# Pooling Methods
POOLING_METHODS = {
    "gap": pooling.GlobalAveragePooling,
    "rap": pooling.RankAveragePooling,
    "ins": pooling.MILInstancePooling,
    "atte": pooling.MILAttentionPooling,
    "ratte": pooling.RankWeightPooling,
    "addi": pooling.MILAdditivePooling,
    "raddi": pooling.MILRankAdditivePooling,
    "conj": pooling.MILConjunctivePooling,
    "rconj": pooling.MILRankConjunctivePooling
}

def get_gpu_device_for_os() -> torch.device:
    if sys.platform == "darwin":
        return torch.device("mps")
    elif sys.platform == "linux":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("Cuda GPU device not found on Linux.")
    raise NotImplementedError("GPU support not configured for platform {:s}".format(sys.platform))


def cross_entropy_criterion(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    loss: float = nn.CrossEntropyLoss()(predictions, targets.long())
    return loss


def set_logger(log_path):
    _logger = logging.getLogger('IHEMOMIL')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%H:%M:%S')

    class TqdmHandler(logging.StreamHandler):
        def __init__(self, formatter):
            logging.StreamHandler.__init__(self)
            self.setFormatter(formatter)

        def emit(self, record):
            msg = self.format(record)
            tqdm.write(msg)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)
    _logger.addHandler(TqdmHandler(fmt))


def data_selecter(dataset_name):
    if dataset_name == "web_traffic":
        from ihemomil.data.web_traffic_dataset import WebTrafficDataset
        train_dataset = WebTrafficDataset("train")
        test_dataset = WebTrafficDataset("test")
    elif dataset_name.rpartition('_')[0] == "mp_ppg":
        from ihemomil.data.morppertur_ppg_dataset import MorpperturPPGDataset
        train_dataset = MorpperturPPGDataset(split="train", name=dataset_name.rpartition('_')[-1])
        test_dataset = MorpperturPPGDataset(split="test", name=dataset_name.rpartition('_')[-1])
    elif dataset_name == "bpf":
        train_dataset = PulseDBPPGDataset(split="train", name=dataset_name.upper())
        test_dataset = PulseDBPPGDataset(split="test", name=dataset_name.upper())
    elif dataset_name == "mimic_af":
        train_dataset = PERFormAFDATASET(split="train", name="MIMIC_PERForm_AF")
        test_dataset = PERFormAFDATASET(split="test", name="MIMIC_PERForm_AF")
    elif dataset_name == "dalia":
        train_dataset = DaLiADATASET(split="train", name="PPG_DaLiA")
        test_dataset = DaLiADATASET(split="test", name="PPG_DaLiA")
    elif dataset_name == "simhf3k_LR":
        train_dataset = SimHF3KDataset(split="train", name="Radial_L")
        test_dataset = SimHF3KDataset(split="test", name="Radial_L")
    elif dataset_name == "simhf3k_LT":
        train_dataset = SimHF3KDataset(split="train", name="Radial_T")
        test_dataset = SimHF3KDataset(split="test", name="Radial_T")

    logger.info("Dataset: {}".format(dataset_name))
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    logger.info("Test dataset size: {}".format(len(test_dataset)))
    logger.info("Number of classes: {}".format(train_dataset.n_clz))
    return train_dataset, test_dataset


class BulidModel(nn.Module):
    def __init__(self, backbone, pooling):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling

    def forward(self, bags, pos=None):
        timestep_embeddings = self.backbone(bags)
        return self.pooling(timestep_embeddings, pos=pos) 
    

def results_log(results_data):
    table = Texttable()
    table.add_rows(results_data)
    table.set_cols_align(["l"] * len(results_data[0]))
    table.set_max_width(0)
    logger.info("\n" + table.draw())
    print("\n" + table.draw())


def plot_conf_mat(conf_mat, save_path=None, clz_names=None, file_name="conf_mat"):
    if clz_names == "web_traffic":
        clz_names = WEBTRAFFIC_CLZ_NAMES
    if clz_names.rpartition('_')[0] == "mp_ppg":
        clz_names = MORPPERTURPPG_CLZ_NAMES
    if clz_names == "mimic_af":
        clz_names = MIMIC_AF_CLZ_NAMES
    if clz_names == "dalia":
        clz_names = ACTIVITY_CLZ_NAMES
    if clz_names == "bpf":
        clz_names = PulseDBBPF_CLZ_NAMES
    if clz_names == "simhf3k_LR":
        clz_names = PERTUR_CLZ_NAMES
    if clz_names == "simhf3k_LT":
        clz_names = PERTUR_CLZ_NAMES

    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    norm_conf_mat = normalize(conf_mat, p=1, dim=1)
    sns.heatmap(
        norm_conf_mat,
        ax=axis,
        fmt=".3f",
        cmap="BuPu",
        cbar_kws={"shrink": 0.7},
        vmin=0,
        vmax=torch.max(norm_conf_mat),
        annot=True,
    )
    axis.set_aspect("equal")
    axis.tick_params(
        axis="x",
        which="both",
        top=False,
        bottom=False,
        labeltop=False,
        labelbottom=True,
    )
    axis.tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=True,
        labelright=False,
        labelrotation=0,
    )
    tick_labels = [
        "Perturbation {:d}:\n{:s}".format(idx, clz_name.capitalize()) for idx, clz_name in enumerate(clz_names)
    ]
    axis.set_xticklabels(tick_labels, rotation=45, ha="right")
    axis.set_yticklabels(tick_labels)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")

    if save_path is not None:
        save_path = os.path.join(save_path, file_name+".png")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        logger.info("Saved confusion matrix to {:s}".format(save_path))

