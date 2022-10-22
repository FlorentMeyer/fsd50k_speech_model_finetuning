from typing import Union, Optional, NoReturn

import pytorch_lightning as pl
import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel


class EmbedderHF(nn.Module):
    """Defines a HuggingFace embedder (e.g. Wav2Vec2) from pre-trained weights."""

    def __init__(self, model_name: PreTrainedModel, hubpath_weights: str):
        super().__init__()

        self.module = model_name.from_pretrained(hubpath_weights)

    def forward(self, xs: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        out = self.module(
            input_values=xs,
            attention_mask=padding_masks,
            output_hidden_states=True,
        )

        return torch.stack(out.hidden_states[1:])  # (L,N,T,C)


class Unfreeze(pl.callbacks.BaseFinetuning):
    """Allows unfreezing components of the embedder at given epochs for fine-tuning."""

    def __init__(self):
        super().__init__()

    def freeze_before_training(self, pl_module):
        embedder = pl_module.model["embedder"].module

        self.freeze(embedder)
        embedder.feature_extractor._requires_grad = False

    def finetune_function(
        self, pl_module, current_epoch: int, optimizer, optimizer_idx
    ):
        embedder = pl_module.model["embedder"].module

        unfreeze = [
            k for k, v in pl_module.hparams.unfreeze.items() if v == current_epoch
        ]
        modules = set()

        for u in unfreeze:
            for n, _ in embedder.named_parameters():
                if u in n:
                    modules.add(_get_module_from_str(n, embedder))

        self.unfreeze_and_add_param_group(
            modules=modules,
            optimizer=optimizer,
        )

        # Copy HuggingFace internal unfreezing behaviour
        names = []
        for u in unfreeze:
            for n, _ in embedder.named_parameters():
                if u in n:
                    names.append(n)
        for n in names:
            if "feature_extractor" in n:
                embedder.feature_extractor._requires_grad = True
                break


class EmbeddingsMerger(nn.Module):
    """
    Taking embeddings of shape (L,N,T,C) output by the embedder,
    merges/reduces them along the Time and Layer dimensions to allow for use
    by the classifier.
    """

    def __init__(self, red_T: str, red_L: str, Ls: list):
        super().__init__()
        assert red_T in ("mean", "max")
        assert red_L in ("mean", "max")
        self.red_T = red_T
        self.red_L = red_L
        self.Ls = torch.tensor(Ls)

    def forward(self, xs: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        # Reduce Time: (L,N,T,C) -> (L,N,C)
        if self.red_T == "mean":
            xs = torch.mean(xs, dim=2)
        elif self.red_T == "max":
            xs = torch.max(xs, dim=2).values

        # Reduce Layer: (L,N,C) -> (N,C)
        Ls = self.Ls.type_as(xs).long()
        xs = torch.index_select(xs, dim=0, index=Ls)  # (L,N,C) -> (L',N,C)
        if self.red_L == "mean":
            xs = torch.mean(xs, dim=0)
        elif self.red_L == "max":
            xs = torch.max(xs, dim=0).values

        return xs


class Classifier(nn.Module):
    """Defines a simple classifier with normalization and 2 dense layers."""

    def __init__(
        self,
        in_size: int,
        activation: nn.Module,
        hidden_size: int,
        normalization: Optional[nn.Module] = None,
    ):
        super().__init__()
        if normalization:
            self.norm = normalization(in_size)
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.activation = activation()
        self.fc2 = nn.Linear(hidden_size, 200)

    def forward(self, xs: torch.Tensor, padding_masks: torch.Tensor) -> torch.Tensor:
        if self.norm:
            xs = self.norm(xs)
        xs = self.fc1(xs)
        xs = self.activation(xs)
        xs = self.fc2(xs)

        return xs


def _get_module_from_str(submodule_str: str, base_module: nn.Module):
    """
    Used in the unfreezing process.
    Allows specifying a string of a model's (e.g. W2V2) submodule to retrieve
    said submodule as an nn.Module.
    Submodules can consist in nested attributes and/or indices ending in an
    unwanted 'weight'/'bias' attribute (which is a Tensor, not an nn.Module).
    Example:
        submodule_str = "encoder.layers.0.attention.k_proj.weight"
        will return the module encoder.layers.0.attention.k_proj
    """
    attrs = submodule_str.split(".")[:-1]
    for _ in range(len(attrs)):
        attr = attrs.pop(0)
        try:
            base_module = base_module[int(attr)]
        except:
            base_module = base_module.__getattr__(attr)
    return base_module


class EmbedderClassifier(pl.LightningModule):
    def __init__(
        self,
        embedder_cls,
        embedder_prm,
        embeddings_merger_cls,
        embeddings_merger_prm,
        classifier_cls,
        classifier_prm,
        loss_cls,
        loss_prm,
        optimizer_cls,
        optimizer_prm,
        unfreeze,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.ModuleDict(
            {
                "embedder": embedder_cls(**embedder_prm),
                "embeddings_merger": embeddings_merger_cls(**embeddings_merger_prm),
                "classifier": classifier_cls(**classifier_prm),
            }
        )

        self.loss_fx = loss_cls(**loss_prm)

        self._check_unfreeze()

    def _check_unfreeze(self) -> NoReturn:
        """
        Makes sure unfreezing will work during training by checking
        beforehand whether the embedder layers to unfreeze match
        the model's layers and correspond to instances of nn.Module.
        """
        embedder = self.model["embedder"].module

        # Vérifier que tous les params à geler existent
        for u in self.hparams.unfreeze:
            check = False
            for n, _ in embedder.named_parameters():
                if u in n and isinstance(_get_module_from_str(n, embedder), nn.Module):
                    check = True
                    break
            if not check:
                raise ValueError(f"{u}: incorrect parameter")

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_cls(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.hparams.optimizer_prm,
        )

        return {"optimizer": optimizer}

    def forward(self, xs, padding_masks):
        for part in self.model.values():
            xs = part(xs, padding_masks)
        return xs

    def training_step(self, batch, batch_idx):
        xs, padding_masks, ys_true = (
            batch["xs"],
            batch.get("padding_masks"),
            batch["ys_true"],
        )
        logits = self(xs, padding_masks)
        loss = self.loss_fx(logits, ys_true.to(torch.float32))
        self.log("train_loss", loss, on_step=None, on_epoch=False)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Allows logging the very 1st step
        for more complete loss curves."""
        if batch_idx == 0 and self.current_epoch == 0:
            self.logger.log_metrics(self.trainer.logged_metrics, step=0)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        xs, padding_masks, ys_true = (
            batch["xs"],
            batch.get("padding_masks"),
            batch["ys_true"],
        )
        logits = self(xs, padding_masks)
        loss = self.loss_fx(logits, ys_true.to(torch.float32))
        metrics = {"val_loss": loss}
        self.log_dict(metrics, on_step=None, on_epoch=True, sync_dist=False)

    def predict_step(self, batch, batch_idx):
        xs, padding_masks, ys_true, file_ids = (
            batch["xs"],
            batch.get("padding_masks"),
            batch["ys_true"],
            batch["file_ids"],
        )
        logits = self(xs, padding_masks)
        logits = torch.sigmoid(logits)  # BCEWithLogitsLoss includes Sigmoid
        return {
            "file_ids": file_ids.cpu(),
            "logits": logits.cpu(),
            "ys_true": ys_true.cpu(),
        }
