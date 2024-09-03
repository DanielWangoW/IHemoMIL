import copy
from typing import Callable, Dict, Tuple, Optional, Union, List, cast

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    balanced_accuracy_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from ihemomil.data.mil_tsc_dataset import MILTSCDataset
from ihemomil.interpretability_metrics import calculate_aopcr, calculate_ndcg_at_n
from ihemomil.utils import custom_tqdm, cross_entropy_criterion

writer = SummaryWriter()

class IHemoMILModel(nn.Module):
    def __init__(self, name: str, device: torch.device, n_classes: int, net: nn.Module):
        super().__init__()
        self.name = name
        self.device = device
        self.n_classes = n_classes
        self.net = net.to(self.device)

    def fit(
        self,
        train_dataset: MILTSCDataset,
        batch_size: int = 16,
        n_epochs: int = 1500,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        criterion: Callable = cross_entropy_criterion,
    ) -> None:

        batch_size = min(len(train_dataset) // 5, batch_size)
        batch_size = max(batch_size, 2)
        torch_train_dataloader = train_dataset.create_dataloader(shuffle=True, batch_size=batch_size)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        best_net = None
        best_loss = np.Inf
        for _ in custom_tqdm(range(n_epochs), desc="Training model"):
            self.net.train()
            for batch in torch_train_dataloader:
                bags = batch["bags"]
                targets = batch["targets"].to(self.device)
                optimizer.zero_grad()
                model_out = self(bags)
                loss = criterion(model_out["bag_logits"], targets)
                loss.backward()
                optimizer.step()
            self.net.eval()
            epoch_train_results = self.evaluate(train_dataset, criterion)
            writer.add_scalar('Accuracy/train', epoch_train_results["acc"], _)
            writer.add_scalar('AUROC/train', epoch_train_results["auroc"], _)
            writer.add_scalar('Loss/train', epoch_train_results["loss"], _)
            epoch_loss = epoch_train_results["loss"]
            if epoch_loss < best_loss:
                best_net = copy.deepcopy(self.net)
                best_loss = epoch_loss
                if epoch_loss == 0:
                    print("Training finished - early stopping (zero loss)")
                    break
        if best_net is not None:
            self.net = best_net
        else:
            raise ValueError("Best net not set during training - shouldn't be here so something has gone wrong!")

    def evaluate(
        self,
        dataset: MILTSCDataset,
        criterion: Callable = cross_entropy_criterion,
    ) -> Dict:
        all_bag_logits_list = []
        all_targets_list = []
        dataloader = dataset.create_dataloader(batch_size=16)
        with torch.no_grad():
            for batch in dataloader:
                bags = batch["bags"]
                targets = batch["targets"]
                model_out = self(bags)
                bag_logits = model_out["bag_logits"]
                all_bag_logits_list.append(bag_logits.cpu())
                all_targets_list.append(targets)
        all_bag_logits = torch.cat(all_bag_logits_list)
        all_targets = torch.cat(all_targets_list)
        all_pred_probas = torch.softmax(all_bag_logits, dim=1)
        if all_pred_probas.shape[1] == 2:
            all_pred_probas = all_pred_probas[:, 1]
        _, all_pred_clzs = torch.max(all_bag_logits, dim=1)
        loss = criterion(all_bag_logits, all_targets).item()
        acc = accuracy_score(all_targets.long(), all_pred_clzs)
        bal_acc = balanced_accuracy_score(all_targets.long(), all_pred_clzs)
        auroc = roc_auc_score(all_targets, all_pred_probas, multi_class="ovo", average="weighted")
        conf_mat = torch.as_tensor(confusion_matrix(all_targets, all_pred_clzs), dtype=torch.float)


        all_results = {
            "loss": loss,
            "acc": acc,
            "bal_acc": bal_acc,
            "auroc": auroc,
            "conf_mat": conf_mat,
        }
        return all_results

    def evaluate_interpretability(
        self,
        dataset: MILTSCDataset,
    ) -> Tuple[float, Optional[float]]:
        all_aopcrs = []
        all_ndcgs = []
        dataloader = dataset.create_dataloader(batch_size=1024)
        with torch.no_grad():
            for batch in custom_tqdm(dataloader, leave=False):
                bags = batch["bags"]
                batch_targets = batch["targets"]
                # Calculate AOPCR for batch
                batch_aopcr, _, _ = calculate_aopcr(self, bags, verbose=False)
                all_aopcrs.extend(batch_aopcr.tolist())
                # Calculate NDCG@n for batch if instance targets are present
                if "instance_targets" in batch:
                    batch_instance_targets = batch["instance_targets"]
                    all_instance_importance_scores = self.interpret(self(bags))
                    for bag_idx, bag in enumerate(bags):
                        target = batch_targets[bag_idx]
                        instance_targets = batch_instance_targets[bag_idx]
                        ndcg = calculate_ndcg_at_n(
                            all_instance_importance_scores[bag_idx, target],
                            instance_targets,
                        )
                        all_ndcgs.append(ndcg)
        avg_aopcr = np.mean(all_aopcrs)
        avg_ndcg = float(np.mean(all_ndcgs)) if len(all_ndcgs) > 0 else None
        return float(avg_aopcr), avg_ndcg

    def interpret(self, model_out: Dict) -> torch.Tensor:
        return model_out["interpretation"]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.net.parameters())

    def save_weights(self, path: str) -> None:
        print("Saving model to {:s}".format(path))
        torch.save(self.net.to("cpu").state_dict(), path)
        # Ensure net is back on original device
        self.net.to(self.device)

    def load_weights(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

    def forward(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        bags, is_unbatched_bag = self._reshape_bag_input(bag_input)
        model_output = self._internal_forward(bags, bag_instance_positions)
        if is_unbatched_bag:
            unbatched_model_output = {}
            for key, value in model_output.items():
                unbatched_model_output[key] = value[0]
            return unbatched_model_output
        return model_output

    def _reshape_bag_input(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], bool]:
        reshaped_input: Union[torch.Tensor, List[torch.Tensor]]
        if torch.is_tensor(bag_input):
            bag_input = cast(torch.Tensor, bag_input)
            input_shape = bag_input.shape
            if len(input_shape) == 2:
                reshaped_input = [bag_input]
                is_unbatched = True
            elif len(input_shape) == 3:
                reshaped_input = bag_input
                is_unbatched = False
            else:
                raise NotImplementedError("Cannot process MIL model input with shape {:}".format(input_shape))
        elif isinstance(bag_input, list):
            reshaped_input = bag_input
            is_unbatched = False
        else:
            raise ValueError("Invalid model input type {:}".format(type(bag_input)))
        return reshaped_input, is_unbatched

    def _internal_forward(
        self, bags: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        if isinstance(bags, list):
            bags = torch.stack(bags)
        bags = bags.to(self.device)
        bags = bags.transpose(1, 2)
        return self.net(bags, bag_instance_positions)

    def __call__(
        self, bag_input: Union[torch.Tensor, List[torch.Tensor]], bag_instance_positions: Optional[torch.Tensor] = None
    ) -> Dict:
        return self.forward(bag_input, bag_instance_positions=bag_instance_positions)
