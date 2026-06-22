import copy
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ihemomil.data.mil_tsc_dataset import MILTSCDataset
from ihemomil.interpretability_metrics import calculate_aopcr, calculate_ndcg_at_n
from ihemomil.utils import cross_entropy_criterion, custom_tqdm

BagInput = Union[torch.Tensor, List[torch.Tensor]]

# A single TensorBoard writer is shared across runs in this process.
_summary_writer = SummaryWriter()


class IHemoMILModel(nn.Module):
    """Training / evaluation harness around a backbone+pooling network in the
    IHemoMIL framework. The wrapped network is stored under ``self.net`` so that
    persisted state dicts stay flat and checkpoint-compatible."""

    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module):
        super().__init__()
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.net = net.to(self.device)

    # -- training ----------------------------------------------------------
    def fit(
        self,
        train_dataset: MILTSCDataset,
        batch_size: int = 16,
        n_epochs: int = 1500,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        criterion: Callable = cross_entropy_criterion,
    ) -> None:
        loader = train_dataset.create_dataloader(
            shuffle=True, batch_size=self._resolve_batch_size(train_dataset, batch_size)
        )
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        champion: Optional[nn.Module] = None
        champion_loss = np.Inf
        for epoch in custom_tqdm(range(n_epochs), desc="Training model"):
            self._run_train_epoch(loader, optimizer, criterion)

            self.net.eval()
            stats = self.evaluate(train_dataset, criterion)
            self._record_scalars(stats, epoch)

            if stats["loss"] < champion_loss:
                champion_loss = stats["loss"]
                champion = copy.deepcopy(self.net)
                if champion_loss == 0:
                    print("Zero training loss reached; halting optimisation early.")
                    break

        if champion is None:
            raise ValueError("No best snapshot was captured while fitting - aborting.")
        self.net = champion

    @staticmethod
    def _resolve_batch_size(dataset: MILTSCDataset, requested: int) -> int:
        """Clamp the requested batch size to at most a fifth of the dataset and
        at least two samples."""
        return max(2, min(len(dataset) // 5, requested))

    def _run_train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable,
    ) -> None:
        """Run one full pass of gradient updates over ``loader``."""
        self.net.train()
        for batch in loader:
            targets = batch["targets"].to(self.device)
            optimizer.zero_grad()
            predictions = self(batch["bags"])["bag_logits"]
            batch_loss = criterion(predictions, targets)
            batch_loss.backward()
            optimizer.step()

    @staticmethod
    def _record_scalars(stats: Dict, step: int) -> None:
        """Mirror the headline epoch metrics into TensorBoard."""
        for tag, key in (("Accuracy", "acc"), ("AUROC", "auroc"), ("Loss", "loss")):
            _summary_writer.add_scalar("{}/train".format(tag), stats[key], step)

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        dataset: MILTSCDataset,
        criterion: Callable = cross_entropy_criterion,
    ) -> Dict:
        """Compute loss and classification metrics over an entire dataset."""
        logit_chunks = []
        target_chunks = []
        with torch.no_grad():
            for batch in dataset.create_dataloader(batch_size=16):
                logit_chunks.append(self(batch["bags"])["bag_logits"].cpu())
                target_chunks.append(batch["targets"])

        logits = torch.cat(logit_chunks)
        targets = torch.cat(target_chunks)
        predicted_classes = logits.argmax(dim=1)

        probabilities = torch.softmax(logits, dim=1)
        if probabilities.shape[1] == 2:
            probabilities = probabilities[:, 1]

        int_targets = targets.long()
        return {
            "loss": criterion(logits, targets).item(),
            "acc": accuracy_score(int_targets, predicted_classes),
            "bal_acc": balanced_accuracy_score(int_targets, predicted_classes),
            "auroc": roc_auc_score(targets, probabilities, multi_class="ovo", average="weighted"),
            "conf_mat": torch.as_tensor(
                confusion_matrix(targets, predicted_classes), dtype=torch.float
            ),
        }

    def evaluate_interpretability(
        self,
        dataset: MILTSCDataset,
    ) -> Tuple[float, Optional[float]]:
        """Score interpretation quality: AOPCR always, plus NDCG@n when the
        dataset carries per-instance ground-truth labels."""
        aopcr_scores: List[float] = []
        ndcg_scores: List[float] = []
        with torch.no_grad():
            for batch in custom_tqdm(dataset.create_dataloader(batch_size=1024), leave=False):
                bags = batch["bags"]
                batch_aopcr, _, _ = calculate_aopcr(self, bags, verbose=False)
                aopcr_scores.extend(batch_aopcr.tolist())
                if "instance_targets" not in batch:
                    continue
                ndcg_scores.extend(self._instance_ndcgs(bags, batch))

        mean_aopcr = float(np.mean(aopcr_scores))
        mean_ndcg = float(np.mean(ndcg_scores)) if ndcg_scores else None
        return mean_aopcr, mean_ndcg

    def _instance_ndcgs(self, bags, batch: Dict) -> List[float]:
        """NDCG@n of the predicted importance ranking against instance labels,
        one value per bag in the batch."""
        importance = self.interpret(self(bags))
        labels = batch["targets"]
        instance_labels = batch["instance_targets"]
        scores = []
        for idx, instance_label in enumerate(instance_labels):
            ranked = importance[idx, labels[idx]]
            scores.append(calculate_ndcg_at_n(ranked, instance_label))
        return scores

    def interpret(self, model_out: Dict) -> torch.Tensor:
        return model_out["interpretation"]

    # -- persistence / introspection --------------------------------------

    def num_params(self) -> int:
        return sum(param.numel() for param in self.net.parameters())

    def save_weights(self, path: str) -> None:
        print("Writing network parameters to {:s}".format(path))
        torch.save(self.net.to("cpu").state_dict(), path)
        self.net.to(self.device)

    def load_weights(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    # -- inference ---------------------------------------------------------
    def forward(
        self, bag_input: BagInput, bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """Run the network on one bag or a batch of bags. A lone 2D bag is
        treated as a batch of one and unwrapped on the way out."""
        batched_bags, was_single = self._reshape_bag_input(bag_input)
        outputs = self._run_network(batched_bags, bag_instance_positions)
        if was_single:
            return {name: tensor[0] for name, tensor in outputs.items()}
        return outputs

    def _reshape_bag_input(self, bag_input: BagInput) -> Tuple[BagInput, bool]:
        """Normalise the accepted bag input forms into a batched collection.

        * 2D tensor (single bag)            -> ([bag], True)
        * 3D tensor (already batched)       -> (tensor, False)
        * list of 2D tensors (batched)      -> (list, False)
        """
        if isinstance(bag_input, list):
            return bag_input, False
        if torch.is_tensor(bag_input):
            tensor = cast(torch.Tensor, bag_input)
            rank = tensor.dim()
            if rank == 2:
                return [tensor], True
            if rank == 3:
                return tensor, False
            raise NotImplementedError(
                "Unsupported bag tensor rank for shape {:}".format(tuple(tensor.shape))
            )
        raise ValueError("Unrecognised bag input container {:}".format(type(bag_input)))

    def _run_network(
        self, bags: BagInput, bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        """Move bags onto the model device, lay them out as (batch, channels,
        timesteps), and forward them through the wrapped network."""
        if isinstance(bags, list):
            bags = torch.stack(bags)
        bags = bags.to(self.device).transpose(1, 2)
        return self.net(bags, bag_instance_positions)

    def __call__(
        self, bag_input: BagInput, bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        return self.forward(bag_input, bag_instance_positions=bag_instance_positions)
