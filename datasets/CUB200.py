import os
import json
import pickle
import random
import math
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json, write_json, mkdir_if_missing

from sklearn.model_selection import train_test_split


@DATASET_REGISTRY.register()
class CUB200(DatasetBase):
    """
    CUB-200-2011 dataset loader following the same conventions as OxfordPets sample.
    Expects the extracted folder CUB_200_2011 with:
      - images/...
      - images.txt
      - image_class_labels.txt
      - train_test_split.txt
    """

    dataset_dir = "CUB_200_2011"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.split_path = os.path.join(self.dataset_dir, "cub_200_2011_splits.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = self.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = self.build_splits_from_meta(val_frac=0.1)  # change val_frac if desired
            self.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, val, test = self.subsample_classes(train, val, test, subsample=subsample)

        super().__init__(train_x=train, val=val, test=test)

    # -----------------------
    # Helpers to read meta
    # -----------------------
    def _read_twocol(self, path):
        """
        Read mapping files like images.txt (id path) or image_class_labels.txt (id label).
        Returns dict[int] -> str
        """
        d = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                key = int(parts[0])
                val = " ".join(parts[1:])
                d[key] = val
        return d

    def build_splits_from_meta(self, val_frac=0.1):
        """
        Build train/val/test lists from CUB meta files:
          - images.txt  (id path)
          - image_class_labels.txt (id class_id)
          - train_test_split.txt (id is_train: 1 train, 0 test)
        Creates a stratified val set from the training portion.
        Returns lists of Datum objects (train, val, test).
        """
        images_txt = os.path.join(self.dataset_dir, "images.txt")
        labels_txt = os.path.join(self.dataset_dir, "image_class_labels.txt")
        split_txt  = os.path.join(self.dataset_dir, "train_test_split.txt")

        images_map = self._read_twocol(images_txt)           # id -> relative path under images/
        labels_map = {k: int(v) for k, v in self._read_twocol(labels_txt).items()}  # id -> class_id (1-based)
        split_map  = {k: int(v) for k, v in self._read_twocol(split_txt).items()}   # id -> is_train (1/0)

        # Build Datum list
        all_samples = []
        for img_id, rel_path in images_map.items():
            class_id = labels_map[img_id]          # 1..200
            is_train = split_map.get(img_id, 0) == 1
            # path on disk
            impath = os.path.join(self.image_dir, rel_path)
            # class name: strip numeric prefix from folder name, e.g. "001.Black_footed_Albatross" -> "Black_footed_Albatross"
            class_folder = rel_path.split("/", 1)[0] if "/" in rel_path else rel_path
            classname = class_folder.split(".", 1)[1] if "." in class_folder else class_folder
            # Convert to 0-based label (common in your code). Change if you want 1-based.
            label0 = class_id - 1
            all_samples.append({"impath": impath, "label": label0, "classname": classname, "is_train": is_train})

        train_samples = [s for s in all_samples if s["is_train"]]
        test_samples  = [s for s in all_samples if not s["is_train"]]

        # create a stratified val set out of train_samples
        if val_frac > 0:
            train_paths = [s["impath"] for s in train_samples]
            train_labels = [s["label"] for s in train_samples]
            tr_paths, val_paths, tr_idx, val_idx = train_test_split(
                train_paths, list(range(len(train_paths))),
                test_size=val_frac, random_state=42, stratify=train_labels
            )
            new_train = [Datum(impath=train_samples[i]["impath"],
                               label=train_samples[i]["label"],
                               classname=train_samples[i]["classname"]) for i in tr_idx]
            val_list =  [Datum(impath=train_samples[i]["impath"],
                               label=train_samples[i]["label"],
                               classname=train_samples[i]["classname"]) for i in val_idx]
        else:
            new_train = [Datum(impath=s["impath"], label=s["label"], classname=s["classname"]) for s in train_samples]
            val_list = []

        test_list = [Datum(impath=s["impath"], label=s["label"], classname=s["classname"]) for s in test_samples]

        print(f"CUB200: Train={len(new_train)}, Val={len(val_list)}, Test={len(test_list)}")
        return new_train, val_list, test_list

    # -----------------------
    # save / read split (same format as OxfordPets)
    # -----------------------
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, "")
                if impath.startswith("/") or impath.startswith(os.sep):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out

        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {"train": train, "val": val, "test": test}
        write_json(split, filepath)
        print(f"Saved split to {filepath}")

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        val = _convert(split["val"])
        test = _convert(split["test"])

        return train, val, test

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group
        represents base classes while the second group represents
        new classes.

        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(
                    impath=item.impath,
                    label=relabeler[item.label],
                    classname=item.classname
                )
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output
