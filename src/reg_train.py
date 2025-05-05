import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
import timm
from model import LitClassifier
import numpy as np

# We reuse the transform and splitting utilities from curriculum_data.
from curriculum_data import get_transform, stratified_split, get_c_scores
from torchvision import datasets
from torch.utils.data import DataLoader


def get_regular_dataloaders(
    dataset_name, batch_size, val_ratio=0.2, model_name=None, val_indices_path=None
):
    """
    Returns standard training and validation dataloaders.
    For CIFAR100, we use a stratified split of the official train set.
    For ImageNet, we use the official train/val directories.
    """
    transform = get_transform(model_name)
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower == "cifar100":
        dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        if val_indices_path and os.path.exists(val_indices_path):
            print("Loading val indices...")
            idx_file = np.load(val_indices_path)
            train_idx = np.sort(idx_file["train_idx"])
            val_idx = np.sort(idx_file["val_idx"])
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            val_dataset = torch.utils.data.Subset(dataset, val_idx)
        else:
            # fresh stratified split (old behaviour)
            difficulties = get_c_scores(dataset_name)["scores"]
            train_dataset, _, val_dataset = stratified_split(
                dataset, difficulties, val_ratio=val_ratio, random_state=42
            )
    elif dataset_name_lower == "imagenet":
        train_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/train", transform=transform
        )
        val_dataset = datasets.ImageFolder(
            root="/datashare/ImageNet/ILSVRC2012/val", transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


def train_model(
    model,
    train_loader,
    val_loader,
    logger,
    checkpoint_callbacks,
    max_epochs,
    lr,
    optimizer,
    weight_decay,
):
    """
    Wraps the model in a Lightning module and starts training.
    Here we assume that regular training uses the staged variant and simple cross-entropy loss.
    """
    lit_model = LitClassifier(
        model,
        lr=lr,
        num_classes=args.num_classes,
        optimizer=optimizer,
        weight_decay=weight_decay,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=checkpoint_callbacks,
        reload_dataloaders_every_n_epochs=1,
    )
    trainer.fit(lit_model, train_loader, val_loader)


def main(args):
    logger = None
    if args.use_comet:
        logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=os.environ.get("COMET_PROJECT_NAME"),
            workspace="nourshaheen",
            experiment_name=args.experiment_name,
        )
        logger.log_hyperparams(vars(args))

    # Create the directory for saving checkpoints.
    save_path = os.path.join(args.save_path, f"{args.experiment_name}_checkpoints")
    os.makedirs(save_path, exist_ok=True)

    # Define a checkpoint callback.
    performance_checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        dirpath=save_path,
        filename="top5-model-{epoch:02d}-{val_f1:.5f}",
        save_top_k=5,
        mode="max",
    )

    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="model-{epoch:02d}-{val_f1:.5f}",
        every_n_epochs=args.every_k,
        save_top_k=-1,
        save_last=True,
    )

    # Create the regular train and validation dataloaders.
    train_loader, val_loader = get_regular_dataloaders(
        args.dataset,
        args.batch_size,
        model_name=args.model_name,
        val_indices_path=args.val_indices_path,
    )

    # Create the model from timm. Use pretrained weights if available.
    model = timm.create_model(
        args.model_name, num_classes=args.num_classes, pretrained=True
    )

    print("Starting training with regular (non-curriculum) experiment...")
    train_model(
        model,
        train_loader,
        val_loader,
        logger,
        [performance_checkpoint_callback, periodic_checkpoint_callback],
        args.max_epochs,
        args.lr,
        args.optimizer,
        args.weight_decay,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regular Training Experiment")

    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        help="Dataset to use; CIFAR100 or imagenet",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet18",
        help="Model architecture (e.g., resnet18, resnet50, etc.)",
    )
    parser.add_argument(
        "--num_classes", type=int, default=100, help="Number of classes in the dataset"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd", "adagrad", "adamw"],
        help="Optimizer to use",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Weight decay (L2 regularization strength)",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="A unique name for the experiment (used for logging)",
    )
    parser.add_argument(
        "--use_comet", action="store_true", help="Enable Comet logging if set"
    )
    parser.add_argument(
        "--val_indices_path",
        type=str,
        default="split/cifar100_val_split.npz",
        help="Path to .npz file with pre‑computed train/val indices",
    )
    parser.add_argument(
        "-k", "--every_k", type=int, default=3, help="Save model every k epochs"
    )

    args = parser.parse_args()
    main(args)
