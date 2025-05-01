import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Precision, Recall, F1Score

def continuous_loss(logits, cluster_targets, coarse_mapping):
    """
    Computes the continuous curriculum loss as defined in the paper.
    
    For each sample i, with target cluster c = cluster_targets[i],
    the loss is:
        L_i = - log( sum_{j in coarse_mapping[c]} exp( logits[i,j] ) )
    and the final loss is the mean over the batch.
    
    This formulation marginalizes out the unobserved fine-class variable, as described in Section 2.2.
    """
    losses = []
    for i in range(logits.size(0)):
        cluster = int(cluster_targets[i].item())
        # Get the list of fine class indices that belong to the target cluster.
        indices = coarse_mapping.get(cluster, [])
        if len(indices) == 0:
            raise ValueError(f"No fine classes found for cluster {cluster} in the coarse mapping.")
        # Use logsumexp for numerical stability.
        cluster_logits = logits[i, indices]
        loss_i = -torch.logsumexp(cluster_logits, dim=0)
        losses.append(loss_i)
    return torch.stack(losses).mean()

#############################################
# PyTorch Lightning Module for Training
#############################################

class LitClassifier(pl.LightningModule):
    def __init__(self, model, lr=1e-3, num_classes=100, batch_size=64, data_fraction=1.0, curriculum_dataset=None, variant="staged", coarse_mapping=None, optimizer="adam", weight_decay=5e-04):
        """
        variant: 'staged' or 'continuous'. 
          - For the continuous variant, the dataloader is expected to return (data, cluster_target),
            and the loss is computed by marginalizing out the fine classes according to the cluster.
          - For the staged variant, standard cross-entropy loss is used.
        coarse_mapping: A dictionary mapping each coarse cluster label to a list of fine class indices.
            This is required for the continuous variant. For example, for CIFAR100 you can obtain it by:
            invert_mapping(get_cifar_mapping()).
        """
        super(LitClassifier, self).__init__()
        self.model = model
        self.lr = lr
        self.data_fraction = data_fraction  
        self.batch_size = batch_size
        self.variant = variant
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.curriculum_dataset = curriculum_dataset  # may contain additional info for curriculum training.
        # For the continuous variant, we require the coarse_mapping.
        if self.variant == "continuous" and coarse_mapping is None:
            raise ValueError("For continuous variant, you must provide a coarse_mapping.")
        self.coarse_mapping = coarse_mapping
        self.criterion = nn.CrossEntropyLoss()
        self.precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.recall = Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)
        
    def on_train_epoch_start(self):
        # This hook is called at the start of every training epoch.
        try:
            # Access the trainer's train_dataloader and get its dataset length.
            # Note: Use self.trainer.current_epoch since self.current_epoch may not be defined.
            dl_length = len(self.trainer.train_dataloader.dataset)
            print(f"\nEpoch {self.trainer.current_epoch}: Training dataloader length = {dl_length}")
        except Exception as e:
            print(f"\nEpoch {self.trainer.current_epoch}: Unable to determine dataloader length. Error: {e}")


    def training_step(self, batch, batch_idx):
        # For continuous variant, assume batch is (data, cluster_target).
        # For staged variant, assume batch is (data, fine_target).
        if self.variant == "continuous":
            data, cluster_target = batch
            logits = self(data)
            loss = continuous_loss(logits, cluster_target, self.coarse_mapping)
            preds = torch.argmax(logits, dim=1)
            # For reporting, we compare the prediction with cluster_target.
            acc = (preds == cluster_target).float().mean()
        else:
            data, target = batch
            logits = self(data)
            loss = self.criterion(logits, target)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == target).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Always validate on fine labels.
        data, target = batch
        logits = self(data)
        loss = self.criterion(logits, target)
        preds = logits.argmax(dim=1)
        acc = (preds == target).float().mean()
        precision = self.precision(preds, target)
        recall = self.recall(preds, target)
        f1 = self.f1(preds, target)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        self.log("data_fraction", self.data_fraction, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer_map = {
            'adam': torch.optim.Adam,
            'adamw': torch.optim.AdamW,
            'sgd': lambda params, **kwargs: torch.optim.SGD(params, momentum=0.9, **kwargs),
            'adagrad': torch.optim.Adagrad,
        }

        opt_name = self.optimizer.lower()
        if opt_name not in optimizer_map:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer}")

        optimizer_class = optimizer_map[opt_name]
        optimizer = optimizer_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_loss',
            }
        }


    def on_save_checkpoint(self, checkpoint):
        # Add additional state to the checkpoint dictionary.
        checkpoint["data_fraction"] = self.data_fraction
        return checkpoint
