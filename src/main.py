from pytorch_lightning.loggers import CometLogger
import torch
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import timm
from curriculum_data import get_data_curriculum_dataloaders, get_c2f_dataloaders
from model import LitClassifier
from pacing_scheduler import PacingScheduler
from utils import UpdateDataLoaderCallback, seed_everything
import numpy as np
from torch import nn
from mapping import get_cifar_mapping, get_imagenet_mapping, invert_mapping


seed_everything(42)


#############################################
# Training Function Using PyTorch Lightning
#############################################

def train_model(model, train_loader, val_loader, logger=None, checkpoint_callbacks=[], batch_size=64,
                lr=1e-03, num_classes=100, max_epochs=10, curriculum_dataset=None,
                variant="staged", coarse_mapping=None, optimizer="adam", weight_decay=5e-04, device="cuda"):
    print("LOGGER:", logger)
    lit_model = LitClassifier(model, lr=lr, num_classes=num_classes, batch_size=batch_size, optimizer=optimizer, weight_decay=weight_decay, curriculum_dataset=curriculum_dataset,
                              variant=variant, coarse_mapping=coarse_mapping)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        accelerator=device,
        devices=1,
        callbacks=checkpoint_callbacks,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=1.0
    )
    trainer.fit(lit_model, train_loader, val_loader)

#############################################
# Main Function
#############################################

def main(args):
    if args.use_comet:
        comet_logger = CometLogger(
            api_key=os.environ["COMET_API_KEY"],
            project_name=os.environ["COMET_PROJECT_NAME"],
            workspace="nourshaheen",
            experiment_name=args.experiment_name
        )
        comet_logger.log_hyperparams(vars(args))
    else:
        comet_logger = None

    save_path = os.path.join(args.save_path, f"{args.experiment_name}_checkpoints")
    os.makedirs(save_path, exist_ok=True)
    
    # Set up checkpoint callbacks (include data fraction in filename)
    performance_checkpoint_callback = ModelCheckpoint(
        monitor="val_f1",
        dirpath=save_path,
        filename="top5-model-{epoch:02d}-{val_f1:.5f}-{data_fraction:.2f}",
        save_top_k=5,
        mode="max"
    )
    periodic_checkpoint_callback = ModelCheckpoint(
        dirpath=save_path,
        filename="model-{epoch:02d}-{val_f1:.5f}-{data_fraction:.2f}",
        every_n_epochs=args.every_k,
        save_top_k=-1,
        save_last=True
    )

    coarse_mapping = None
    
    #  Model and DataLoader initialization and (if needed) the curriculum dataset based on curriculum type and task stage.
    if args.curriculum_type == "data":
        pacing_scheduler = PacingScheduler(method=args.pacing, alpha0=args.starting_percent, 
                                           step=args.step, lambda_=args.linear_growth, r=args.exponential_growth)
        train_loader, val_loader, curriculum_dataset = get_data_curriculum_dataloaders(args.dataset, args.batch_size, pacing_scheduler, args.model_name, args.balance_batches, val_indices_path=args.val_indices_path)
        update_callback = UpdateDataLoaderCallback(curriculum_dataset, args.batch_size, shuffle=True)
        callbacks_list = [performance_checkpoint_callback, periodic_checkpoint_callback, update_callback]
        model = timm.create_model(args.model_name, num_classes=args.num_classes, pretrained=args.pretrained)
        
    elif args.curriculum_type == "task":
        train_loader, val_loader = get_c2f_dataloaders(args.dataset, args.batch_size, args.task_curr_stage, args.model_name, args.balance_batches, val_indices_path=args.val_indices_path)
        curriculum_dataset = None
        callbacks_list = [performance_checkpoint_callback, periodic_checkpoint_callback]
        if args.task_curr_stage == 1:
            # Stage 1 (coarse): train normally.
            model = timm.create_model(args.model_name, num_classes=args.num_classes, pretrained=args.pretrained)
            # Initialize coarse_mapping if variant is continuous.
            if args.variant == "continuous":
                if args.dataset.lower() == "cifar100":
                    fine_to_coarse = get_cifar_mapping()
                    coarse_mapping = invert_mapping(fine_to_coarse)
                elif args.dataset.lower() == "imagenet":
                    fine_to_coarse = get_imagenet_mapping()
                    coarse_mapping = invert_mapping(fine_to_coarse)
                else:
                    raise ValueError("Unsupported dataset for continuous variant.")
        elif args.task_curr_stage == 2:
            if args.prev_checkpoint and os.path.exists(args.prev_checkpoint):
                print(f"Loading checkpoint from {args.prev_checkpoint} for task curriculum stage 2...")
                checkpoint = torch.load(args.prev_checkpoint)
                # Create the model without pretrained weights because we load the checkpoint.
                state_dict = checkpoint["state_dict"]
                new_state = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_key = k[len("model."):]
                    else:
                        new_key = k
                    new_state[new_key] = v
                model = timm.create_model(args.model_name, num_classes=args.old_classes, pretrained=False)
                model.load_state_dict(new_state, strict=False)
                # For continuous variant, do NOT reinitialize the predictor.
                if args.variant == "continuous":
                    # Build an identity mapping: each fine label maps to a singleton list.
                    coarse_mapping = {i: [i] for i in range(args.num_classes)}
                elif args.variant == "staged":
                    print("Using staged variant: reinitializing the predictor layer.")
                    try:
                        in_features = model.fc.in_features  
                        model.fc = nn.Linear(in_features, args.num_classes)
                    except:
                        in_features = model.head.in_features
                        model.head = nn.Linear(in_features, args.num_classes)
            else:
                raise ValueError("Checkpoint required for task curriculum stage 2.")

    # Test one batch from the DataLoader.
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {np.array(target).shape}")
        output = model(data)
        print(f"Model output shape: {output.shape}")
        break

    print("Starting training with PyTorch Lightning...")
    train_model(model, train_loader, val_loader, comet_logger, callbacks_list, batch_size=args.batch_size, lr=args.lr, 
                num_classes=args.num_classes, max_epochs=args.max_epochs, 
                curriculum_dataset=curriculum_dataset, variant=args.variant, coarse_mapping=coarse_mapping, optimizer=args.optimizer, weight_decay=args.weight_decay, device=args.device)

#############################################
# Main 
#############################################

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run curriculum learning experiments")
   
    ###############################
    # Basic Args
    ###############################
    parser.add_argument('--curriculum_type', type=str, default='data', choices=['data', 'task'], help='Type of curriculum experiment')
    parser.add_argument('--dataset', type=str, default='CIFAR100', help='Dataset to use') # e.g., CIFAR100 or imagenet
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Model and Training Settings
    parser.add_argument('--model_name', type=str, default='resnet18', help='Model architecture to use')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--max_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--balance_batches', action='store_true', help='Balance classes per batch if set')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model on imagenet in Data curriculum or coarse task curriculum')
    parser.add_argument('--old_classes', type=int, default=20, help='Number of old output classes')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adagrad', 'adamw'], help='Optimizer to use')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization strength)')

    ###############################
    # Curriculum Dataset Settings
    ###############################
    parser.add_argument('--pacing', type=str, default='linear', choices=['linear', 'exponential'], help='Pacing function')
    parser.add_argument('-s', '--step', type=int, default=1, help='Number of epochs for which the pacing function remains constant')
    parser.add_argument('-p', '--starting_percent', type=float, default=0.3, help='Fraction of data used in the initial step')
    parser.add_argument('-l', '--linear_growth', type=float, default=0.1, help='Linear growth factor to expand the data used in each step')
    parser.add_argument('-r', '--exponential_growth', type=float, default=1.5, help='Exponential growth factor to expand the data used in each step')
    
    ###############################
    # Curriculum Task Settings
    ###############################
    parser.add_argument('--variant', type=str, default='staged', choices=['staged', 'continuous'], help='Variant used for coarse-to-fine curriculum learning')
    parser.add_argument('--task_curr_stage', type=int, default=1, choices=[1, 2], help='Curriculum stage: coarse (1) or fine (2)')
    
    ###############################
    # Checkpointing and Logging
    ###############################
    parser.add_argument('--prev_checkpoint', type=str, default='', help='Path to previous stage checkpoint')
    parser.add_argument('--save_path', type=str, default='./checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--experiment_name', type=str, required=True, help='Name of the experiment for logging')
    parser.add_argument('--use_comet', action='store_true', help='Enable Comet logging if set')
    parser.add_argument('-k', '--every_k', type=int, default=3, help='Save model every k epochs')
    parser.add_argument('-d', '--device', type=str, default="cuda", help='Device')
    
    parser.add_argument('--val_indices_path', type=str, default='./split/cifar100_val_split.npz', help='NPZ file with pre‑computed train/val indices')

    args = parser.parse_args()
    main(args)
