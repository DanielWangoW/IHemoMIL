"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
SPDX-License-Identifier: Apache-2.0
"""
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from ihemomil.model.ihemomil_model import IHemoMILModel
from ihemomil.utils import custom_tqdm


def calculate_aopcr(
    model: "IHemoMILModel",
    bags: List[torch.Tensor],
    verbose: bool = True,
    stop: float = 0.5,
    step: float = 0.05,
    n_random: int = 3,
    seed: int = 72,
    batch_interpretations: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_model_out = model(bags)
    batch_bag_logits = batch_model_out["bag_logits"].cpu()
    batch_pred_clzs = torch.argmax(batch_bag_logits, dim=1).tolist()
    batch_orig_logits = torch.zeros(len(bags))
    for i in range(len(bags)):
        clz = batch_pred_clzs[i]
        batch_orig_logits[i] = batch_bag_logits[i, clz]
    if batch_interpretations is None:
        all_batch_interpretations = model.interpret(batch_model_out)
        batch_interpretations_list = []
        for batch_idx, clz in enumerate(batch_pred_clzs):
            batch_interpretations_list.append(all_batch_interpretations[batch_idx, clz, :])
        batch_interpretations = torch.stack(batch_interpretations_list)
    aopc, pc = _calculate_aopc(
        model,
        bags,
        batch_orig_logits,
        batch_pred_clzs,
        batch_interpretations,
        stop,
        step,
        verbose,
    )
    r_aopc, r_pc = _calculate_random_aopc(
        model,
        bags,
        batch_orig_logits,
        batch_pred_clzs,
        stop,
        step,
        verbose,
        n_random,
        seed,
    )
    aopcr = aopc - r_aopc
    return aopcr, pc, r_pc


def _calculate_aopc(
    model: "IHemoMILModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    batch_interpretation_scores: torch.Tensor,
    stop: float,
    step: float,
    verbose: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_bags = len(bags)
    n_instances = len(bags[0])
    steps = np.linspace(1, stop, num=math.ceil((1 - stop) / step + 1))
    n_steps = len(steps)
    batch_pc = torch.zeros((n_bags, n_steps))
    batch_pc[:, 0] = batch_orig_logits
    batch_morf = torch.argsort(batch_interpretation_scores.cpu(), descending=True, stable=True)
    for step_idx, step in custom_tqdm(
        enumerate(steps[1:]),
        total=n_steps - 1,
        desc="Computing perturbation curve",
        leave=False,
        disable=not verbose,
    ):
        n_to_remove = int((1 - step) * n_instances)
        perturbed_bags = []
        perturbed_bags_pos = []
        for i in range(n_bags):
            b, p = _create_perturbed_bag(bags[i], batch_morf[i], n_to_remove)
            perturbed_bags.append(b)
            perturbed_bags_pos.append(p)
        with torch.no_grad():
            new_logits = model(perturbed_bags, torch.stack(perturbed_bags_pos))["bag_logits"]
        for batch_idx, clz in enumerate(clzs):
            batch_pc[batch_idx, step_idx + 1] = new_logits[batch_idx, clz].item()
    batch_aopc = torch.zeros(n_bags)
    for k in range(1, n_steps):
        batch_aopc += batch_pc[:, 0] - batch_pc[:, k]
    batch_aopc /= n_steps
    batch_pc -= batch_orig_logits.unsqueeze(1)
    return batch_aopc, batch_pc


def _calculate_random_aopc(
    model: "IHemoMILModel",
    bags: List[torch.Tensor],
    batch_orig_logits: torch.Tensor,
    clzs: List[int],
    stop: float,
    step: float,
    verbose: bool,
    n_repeats: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n_bags = len(bags)
    n_instances = len(bags[0])
    torch.random.manual_seed(seed)
    random_aopcs = []
    random_pcs = []
    for r in range(n_repeats):
        random_interpretation_scores = torch.rand((n_bags, n_instances))
        r_aopc, r_pc = _calculate_aopc(
            model,
            bags,
            batch_orig_logits,
            clzs,
            random_interpretation_scores,
            stop,
            step,
            verbose,
        )
        random_aopcs.append(r_aopc)
        random_pcs.append(r_pc)
    aopc = torch.stack(random_aopcs).mean(dim=0)
    pc = torch.stack(random_pcs).mean(dim=0)
    return aopc, pc


def _create_perturbed_bag(
    bag: torch.Tensor, bag_morf: torch.Tensor, n_to_remove: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = torch.ones(len(bag), dtype=torch.int)
    idxs_to_remove = bag_morf[:n_to_remove]
    mask[idxs_to_remove] = 0
    perturbed_bag = bag[mask == 1]
    perturbed_bag_pos = torch.arange(len(bag))[mask == 1]
    return perturbed_bag, perturbed_bag_pos


def calculate_ndcg_at_n(instance_importance_scores: torch.Tensor, instance_labels: torch.Tensor) -> float:
    n = int((instance_labels == 1).sum().item())
    if n == 0:
        print("Shouldn't happen!")
        raise ValueError("Trying to assess interpretability with no discriminatory instances")
    top_n = torch.topk(instance_importance_scores.to("cpu"), n)[1]
    dcg = 0.0
    norm = 0.0
    for i, order_idx in enumerate(top_n):
        rel = instance_labels[order_idx].item()
        dcg += rel / math.log2(i + 2)
        norm += 1.0 / math.log2(i + 2)
    ndcg = dcg / norm
    return ndcg
