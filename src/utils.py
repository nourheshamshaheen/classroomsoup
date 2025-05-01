from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
import numpy as np
from collections import defaultdict
import random
import torch
import pytorch_lightning as pl


def seed_everything(seed: int = 42):
    # Python built-in random module seed
    random.seed(seed)

    # NumPy seed
    np.random.seed(seed)

    # PyTorch seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensuring deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch Lightning convenience function (if using Lightning)
    pl.seed_everything(seed)


class UpdateDataLoaderCallback(Callback):
    def __init__(self, curriculum_dataset, batch_size, shuffle=True):
        """
        Callback to update the training dataloader each epoch.

        Parameters:
        - curriculum_dataset: The dataset instance that supports updating its selected indices.
        - batch_size: Batch size for the new DataLoader.
        - shuffle: Whether to shuffle the data.
        """
        super().__init__()
        self.curriculum_dataset = curriculum_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_train_epoch_start(self, trainer, pl_module):
        # Use the pacing scheduler to get the new fraction and update dataset indices.
        new_fraction = self.curriculum_dataset.pacing_scheduler.next_epoch()
        self.curriculum_dataset.update_indices()
        pl_module.data_fraction = (
            self.curriculum_dataset.pacing_scheduler.current_fraction
        )
        print(
            f"\nEpoch {trainer.current_epoch}: Updated dataloader with {new_fraction*100:.1f}% of data."
        )


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        """
        dataset: your CurriculumDataset (must return (image, label))
        batch_size: number of samples per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.label_to_indices = defaultdict(list)

        # Build a dict: label -> list of indices
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

    def __iter__(self):
        # Copy of indices for each label
        pools = {
            label: indices.copy() for label, indices in self.label_to_indices.items()
        }

        all_batches = []
        while True:
            batch = []
            # Try to sample uniformly across labels
            labels = random.sample(
                self.labels, k=min(len(self.labels), self.batch_size)
            )
            for label in labels:
                if pools[label]:
                    idx = pools[label].pop()
                    batch.append(idx)
                if len(batch) == self.batch_size:
                    break
            if len(batch) < self.batch_size:
                break  # no more full batches
            all_batches.append(batch)

        # Flatten batches into a long list
        return iter([idx for batch in all_batches for idx in batch])

    def __len__(self):
        return len(self.dataset)
