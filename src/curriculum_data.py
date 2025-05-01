from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from mapping import get_cifar_mapping, get_imagenet_mapping
from utils import BalancedBatchSampler
import os


##########################
# Get A Train DataLoader
#########################
def get_a_train_dataloader(dataset, batch_size, balance_batches):
    if balance_batches:
        sampler = BalancedBatchSampler(dataset, batch_size=batch_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)


#############################################
# Utility Dataset Wrapper for Coarse Labels
#############################################


class CoarseLabelDataset(Dataset):
    """
    Wraps a base dataset and maps each label using mapping_fn.
    """

    def __init__(self, base_dataset, mapping_fn):
        self.base_dataset = base_dataset
        self.mapping_fn = mapping_fn
        # Precompute targets once for use in training and for stratified splitting
        try:
            self.targets = [self.mapping_fn(t) for t in base_dataset.targets]
        except AttributeError:
            self.targets = [
                self.mapping_fn(base_dataset[i][1]) for i in range(len(base_dataset))
            ]

    def __getitem__(self, index):
        image, _ = self.base_dataset[index]
        return image, self.targets[index]

    def __len__(self):
        return len(self.base_dataset)


#############################################
# Curriculum Dataset and Dataloader
#############################################


def get_c_scores(dataset_name):
    if dataset_name == "CIFAR100":
        cscores_path = "src/cscores/cifar100-cscores-orig-order.npz"
    elif dataset_name.lower() == "imagenet":
        cscores_path = "src/cscores/imagenet-cscores-with-filename.npz"
    else:
        raise NotImplementedError(f"cscores not implemented for dataset {dataset_name}")
    npz_file = np.load(cscores_path, allow_pickle=True)
    cscores = dict(npz_file.items())
    return cscores


def stratified_split(
    dataset, difficulties, val_ratio=0.2, random_state=42, save_to=None, load_from=None
):
    """
    Splits a dataset and its corresponding difficulty scores into training and validation subsets
    using a stratified split based on the dataset labels.
    (Used only for datasets like CIFAR100 without an official validation split.)
    """
    try:
        labels = np.array(dataset.targets)
    except AttributeError:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    if load_from and os.path.exists(load_from):
        print("Loading val indices...")
        indices = np.load(load_from)
        train_idx, val_idx = indices["train_idx"], indices["val_idx"]
    else:
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=random_state
        )
        train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
        if save_to:
            np.savez(save_to, train_idx=train_idx, val_idx=val_idx)

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    difficulties = np.array(difficulties)
    train_difficulties = difficulties[train_idx]

    return train_dataset, train_difficulties, val_dataset


class CurriculumDataset(Dataset):
    """
    Orders the training samples by increasing difficulty and selects a fraction
    of the easiest samples based on a pacing scheduler.
    """

    def __init__(self, base_dataset, difficulties, pacing_scheduler):
        self.base_dataset = base_dataset
        self.pacing_scheduler = pacing_scheduler
        self.difficulties = np.array(difficulties)
        if len(self.difficulties) != len(self.base_dataset):
            raise ValueError("Mismatch between number of C‑scores and dataset samples.")
        self.sorted_indices = np.argsort(self.difficulties)
        fraction = self.pacing_scheduler.current_fraction
        num_samples = int(fraction * len(self.base_dataset))
        self.selected_indices = self.sorted_indices[:num_samples]

    def update_indices(self):
        fraction = self.pacing_scheduler.current_fraction
        num_samples = int(fraction * len(self.base_dataset))
        self.selected_indices = self.sorted_indices[:num_samples]
        print(
            f"Updated curriculum dataset: using {num_samples} samples out of {len(self.base_dataset)}."
        )

    def update_indices_using_fraction(self, fraction):
        num_samples = int(fraction * len(self.base_dataset))
        self.selected_indices = self.sorted_indices[:num_samples]
        print(
            f"Updated curriculum dataset: using {num_samples} samples out of {len(self.base_dataset)}."
        )

    def __getitem__(self, index):
        real_index = self.selected_indices[index]
        return self.base_dataset[real_index]

    def __len__(self):
        return len(self.selected_indices)


def get_transform(model=None):
    if model is not None and ("vit" in model.lower() or "swin" in model.lower()):
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # Use ImageNet normalization when using a model pretrained on ImageNet
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([transforms.ToTensor(), normalize])
    return transform


def get_data_curriculum_dataloaders(
    dataset_name,
    batch_size,
    pacing_scheduler,
    model=None,
    balance_batches=False,
    val_indices_path=None,
):
    transform = get_transform(model)

    if dataset_name == "CIFAR100":
        # CIFAR100 has only a train split so we do a manual split.
        dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        difficulties = get_c_scores(dataset_name)["scores"]
        train_dataset, train_difficulties, val_dataset = stratified_split(
            dataset,
            difficulties,
            val_ratio=0.2,
            random_state=42,
            load_from=val_indices_path,
        )
    elif dataset_name.lower() == "imagenet":
        # For ImageNet, load the official train and validation splits.
        train_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/train", transform=transform
        )
        val_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/val", transform=transform
        )
        difficulties = get_c_scores(dataset_name)[
            "scores"
        ]  # Assumes difficulties correspond to the train set.
        # We use the full train_dataset for curriculum learning.
        train_difficulties = difficulties
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    curriculum_dataset = CurriculumDataset(
        train_dataset, train_difficulties, pacing_scheduler
    )

    train_loader = get_a_train_dataloader(
        curriculum_dataset, batch_size, balance_batches
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, curriculum_dataset


def get_c2f_dataloaders(
    dataset_name,
    batch_size,
    task_stage,
    model=None,
    balance_batches=False,
    val_indices_path=None,
):
    """
    Returns training and validation dataloaders for coarse-to-fine curriculum learning.
    For task_stage == 1, coarse labels are used; otherwise fine labels.
    """
    transform = get_transform(model)

    if dataset_name == "CIFAR100":
        # Load CIFAR100 train split normally.
        original_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        if task_stage == 1:
            # Get the hard-coded mapping.
            cifar_fine_to_coarse = (
                get_cifar_mapping()
            )  # This returns a dict mapping fine -> coarse.
            cifar_mapping_fn = lambda fine_label: cifar_fine_to_coarse[fine_label]
            full_dataset = CoarseLabelDataset(
                original_dataset, mapping_fn=cifar_mapping_fn
            )
        else:
            full_dataset = original_dataset
        if val_indices_path and os.path.exists(val_indices_path):
            print("Loading val indices...")
            idx_file = np.load(val_indices_path)
            train_idx = np.sort(idx_file["train_idx"])
            val_idx = np.sort(idx_file["val_idx"])
        else:
            try:
                labels = np.array(full_dataset.targets)
            except AttributeError:
                labels = np.array(
                    [full_dataset[i][1] for i in range(len(full_dataset))]
                )
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
            train_idx, val_idx = np.sort(train_idx), np.sort(val_idx)

        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)

    elif dataset_name.lower() == "imagenet":
        # For ImageNet, load the official train and validation splits.
        train_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/train", transform=transform
        )
        val_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/val", transform=transform
        )
        if task_stage == 1:
            imagenet_fine_to_coarse = (
                get_imagenet_mapping()
            )  # mapping: fine label index -> coarse label (string)
            imagenet_mapping_fn = lambda fine_label: imagenet_fine_to_coarse[fine_label]
            train_dataset = CoarseLabelDataset(
                train_dataset, mapping_fn=imagenet_mapping_fn
            )
            val_dataset = CoarseLabelDataset(
                val_dataset, mapping_fn=imagenet_mapping_fn
            )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    train_loader = get_a_train_dataloader(train_dataset, batch_size, balance_batches)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_testing_dataloader(dataset_name, batch_size, model=None):
    """
    Returns a Test DataLoader.
    """
    transform = get_transform(model)

    dataset_name_l = dataset_name.lower()
    if dataset_name_l == "cifar100":
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset_name_l == "imagenet":
        # ImageNet validation set (2012).  Adjust the path if needed.
        test_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/val", transform=transform
        )
    else:
        raise NotImplementedError(f"Testing not implemented for {dataset_name}")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    print(f"Test set: Using _fine_ labels only for {dataset_name}.")
    return test_loader
