import csv
import os
import os.path as osp
from typing import Iterable

import IPython.display as ipd
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import Resample


class FSD50KDataDict:
    """
    Builds dicts formatted as
    {"file_ids": [], "paths": [], "ys_true": []}
    from a *subset* of FSD50K.
    Holds the token_to_name dict useful for retrieving class names
    from logits after the prediction step.
    Notes:
        - *The dataset structure and content have been altered
        to reduce its size, hence _build_split()'s logic
        should be adapted for the full-size FSD50K.*
        - AVOID contains the IDs of broken files
        (these may have been updated in newer versions of FSD50K).
    """

    NUM_LABELS = 200
    AVOID = {
        "train": [
            "124834",
            "124858",
            "121426",
            "121471",
            "121472",
            "124800",
            "83298",
            "83299",
            "121351",
            "124796",
            "124797",
        ],
        "val": ["397150"],
        "test": ["30303"],
    }

    def __init__(
        self,
        dpath_data: str,
    ):
        self.dpath_data = dpath_data
        self.name_to_token = {}
        self.token_to_name = {}
        self._build_mappings()

    def _build_mappings(self):
        fpath_vocabulary = osp.join(self.dpath_data, "ground_truth", "vocabulary.csv")

        with open(fpath_vocabulary, "r") as file:
            reader = csv.reader(file)
            for line in reader:
                self.name_to_token[line[1]] = int(line[0])
        assert len(self.name_to_token) == self.NUM_LABELS

        self.token_to_name = {t: n for n, t in self.name_to_token.items()}

    def _build_split(self, name: str) -> dict:
        if not name in ("train", "val", "test"):
            raise ValueError
        fpath_gt = osp.join(self.dpath_data, "ground_truth", "labels.csv")
        dpath_data = osp.join(self.dpath_data, name)
        file_ids = []
        paths = []
        labels = []
        file_ids_reduced = [name.strip(".wav") for name in os.listdir(dpath_data)]

        with open(fpath_gt, "r") as file:
            reader = csv.DictReader(file)
            for line in reader:
                file_id = line["fname"]
                if file_id in self.AVOID[name]:
                    continue
                if file_id not in file_ids_reduced:
                    continue
                file_ids.append(torch.tensor(int(file_id)))
                paths.append(osp.join(dpath_data, f"{file_id}.wav"))
                labels.append(line["labels"].split(","))

        return {
            "file_ids": file_ids,
            "paths": paths,
            "ys_true": labels,
        }

    def _multi_hot(self, tokenized_labels: list) -> torch.Tensor:
        labels = torch.tensor(
            tokenized_labels
        ).long()  # LongTensor required for one_hot
        labels = nn.functional.one_hot(labels, num_classes=self.NUM_LABELS).sum(dim=0)

        return labels

    def _tokenize(self, split: dict) -> dict:
        for i, names in enumerate(split["ys_true"]):
            tokenized_labels = [self.name_to_token[n] for n in names]
            multi_hot_labels = self._multi_hot(tokenized_labels)
            split["ys_true"][i] = multi_hot_labels

        return split

    def get_dict(self, name: str) -> dict:
        split = self._build_split(name)
        split = self._tokenize(split)

        return split


def tokens_to_names(ys: list, token_to_name: dict) -> list:
    """
    Converts all tokens of all entries in a list of multi-hot labels
    into their corresponding string class names.
    """
    names = []
    for i, multi_hot in enumerate(ys):
        tokens = np.flatnonzero(multi_hot).tolist()
        names.append([token_to_name[t] for t in tokens])

    return names


def inspect_data(datadict: dict, show_keys: list, samples_indices: Iterable = None):
    """
    Displays a dataframe containing the fields from datadict
    set by show_keys as well as audio players.
    Used for inspecting a dataset/predictions with their audio,
    their labels, the predicted logits, etc.
    """
    if samples_indices is None:
        samples_indices = range(len(datadict[show_keys[0]]))

    datadict = {k: [v[i] for i in samples_indices] for k, v in datadict.items()}
    show_dict = {k: v for k, v in datadict.items() if k in show_keys}
    for k, v in show_dict.items():
        if isinstance(v, np.ndarray):
            show_dict[k] = v.tolist()
    df = pd.DataFrame(show_dict)

    df["audio"] = [
        torchaudio.load(
            fpath,
            normalize=False,
        )
        for fpath in datadict["paths"]
    ]

    df["audio"] = df["audio"].apply(
        lambda x: ipd.Audio(data=x[0], rate=x[1])
        ._repr_html_()
        .replace("\n", "")
        .strip()
    )

    ipd.display(ipd.HTML(df.to_html(escape=False, index=False)))


class _FSD50KDataset(Dataset):
    def __init__(self, split_dict: dict, orig_sr: int, goal_sr: int):
        self.file_ids = split_dict["file_ids"]
        self.paths = split_dict["paths"]
        self.ys_true = split_dict["ys_true"]
        self.transform = None
        if goal_sr != orig_sr:
            self.transform = Resample(orig_freq=orig_sr, new_freq=goal_sr)

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, index):
        file_id = self.file_ids[index]

        x, _ = torchaudio.load(
            self.paths[index],
            normalize=False,
        )  # -> (torch.Tensor(channel, time), int)
        x = x.squeeze()  # (C,T) -> (T)
        x = x.to(torch.float32)
        # Zero-mean and unit-var normalization
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        if self.transform:
            x = self.transform(x)

        y_true = self.ys_true[index]

        return {
            "file_id": file_id,
            "x": x,
            "y_true": y_true,
        }


class CollatorVariableLengths:
    """
    Data collator replacing torch's default one,
    allowing samples of different lengths to be batched together
    by padding to make them as long as the longest sample.
    Pads and masks following the HuggingFace fashion:
    padded values have a 0 mask, unpadded values a 1 mask.
    """

    def __init__(self):
        self.mask = 0
        self.nomask = 1 - self.mask

    def __call__(self, batch: dict) -> dict:
        xs = []
        padding_masks = []

        xs_lens = [item["x"].shape[0] for item in batch]
        # If all equal then no need to pad nor mask
        if all(xl == xs_lens[0] for xl in xs_lens):
            xs = [item["x"] for item in batch]
        else:
            batch_max_len = max(xs_lens)
            for item in batch:
                x_len = item["x"].shape[0]
                len_diff = batch_max_len - x_len
                dim = item["x"].dim()
                pad = (0,) * (1 + (dim - 1) * 2) + (len_diff,)
                xs.append(nn.functional.pad(item["x"], pad))
                padding_mask = nn.functional.pad(
                    torch.full((x_len,), self.nomask), (self.mask, len_diff)
                )
                padding_masks.append(padding_mask)

        xs = torch.stack(xs)
        file_ids = torch.stack([item["file_id"] for item in batch])
        ys_true = torch.stack([item["y_true"] for item in batch])

        out_batch = {"file_ids": file_ids, "xs": xs, "ys_true": ys_true}
        if padding_masks:
            out_batch["padding_masks"] = torch.stack(padding_masks)

        return out_batch


class FSD50KDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        shuffle,
        drop_last,
        num_workers,
        pin_memory,
        datadict_prm,
        dataset_prm,
        collate_cls,
        size_limits=None,
        main_labels=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.datadict = FSD50KDataDict(**datadict_prm)
        self.dataset = _FSD50KDataset
        self.collate_fx = collate_cls()

    def setup(self, stage=None):
        if stage in (None, "fit"):
            train_dict = self.datadict.get_dict("train")
            self.train_set = self.dataset(train_dict, **self.hparams.dataset_prm)
            val_dict = self.datadict.get_dict("val")
            self.val_set = self.dataset(val_dict, **self.hparams.dataset_prm)
        if stage in (None, "predict"):
            test_dict = self.datadict.get_dict("test")
            self.test_set = self.dataset(test_dict, **self.hparams.dataset_prm)

    def teardown(self, stage=None):
        if stage in (None, "fit"):
            del self.train_set
            del self.val_set
        if stage in (None, "predict"):
            del self.test_set

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fx,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fx,
            shuffle=self.hparams.shuffle,
            drop_last=self.hparams.drop_last,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fx,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )


def gather_preds(preds: list) -> dict:
    """
    Takes outputs from pl.trainer.predict() as a list of batches dicts
    to gather into a single dict.
    """
    preds_merged = {k: [] for k in preds[0].keys()}

    for d in preds:
        for k, v in d.items():
            preds_merged[k].append(v)

    for k, v in preds_merged.items():
        preds_merged[k] = torch.cat(v)

    return preds_merged


def get_preds_fpaths(preds: dict) -> dict:
    """Retrieves audio file paths from their IDs."""
    dpath_audio = osp.abspath(osp.join(os.sep, "content", "dataset", "test"))
    preds["paths"] = [
        osp.join(dpath_audio, f"{file_id}.wav") for file_id in preds["file_ids"]
    ]

    return preds


def sort_highest_logits(preds: dict, token_to_name: dict, num_classes: int = 4) -> dict:
    """
    For each of the samples on which a prediction was made,
    ranks the num_classes highest confidence logits
    with their corresponding class names.
    """
    preds = {k: np.asarray(v) for k, v in preds.items()}
    preds[f"logits_{num_classes}_highest"] = [
        [
            (token_to_name[token], f"{logit:.2f}")
            for token, logit in zip(
                np.argsort(logits)[::-1][:num_classes],
                np.sort(logits)[::-1][:num_classes],
            )
        ]
        for logits in preds["logits"]
    ]

    return preds


def get_preds_max_logits_indices(preds: dict) -> list:
    """Ranks samples according to the highest logit they contain."""
    preds = {k: np.asarray(v) for k, v in preds.items()}
    ranked_indices = list(np.argsort(preds["logits"].max(axis=1)))[::-1]

    return ranked_indices
