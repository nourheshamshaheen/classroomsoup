import argparse, re, torch, timm
from curriculum_data import get_testing_dataloader
from torchmetrics import F1Score
from tqdm import tqdm


def load_checkpoint_into_model(checkpoint_path, model_name, num_classes, device):
    """Create the timm model and load weights (handles Lightning prefixes)."""
    model = timm.create_model(model_name, num_classes=num_classes)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")

    state = ckpt.get("state_dict", ckpt)  # Lightning or raw
    stripped = {re.sub(r"^(?:model\.|pl_module\.)+", "", k): v
                for k, v in state.items()}
    model.load_state_dict(stripped, strict=False)
    model.to(device)
    model.eval()
    return model

def evaluate(model, dataloader, device, num_classes):
    """Compute accuracy and macro‑F1 over dataloader."""
    f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="macro")
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, lbls in tqdm(dataloader, desc="Testing", unit="batch"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            logits     = model(imgs)
            preds      = torch.argmax(logits, dim=1)

            # accuracy like in LitClassifier
            correct += (preds == lbls).sum().item()
            total   += lbls.numel()

            # update F1 on CPU tensors (TorchMetrics default device is CPU)
            f1_metric.update(preds.cpu(), lbls.cpu())

    acc = correct / total
    f1  = f1_metric.compute().item()
    return acc, f1


def main(args):
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    print(f"Using device: {device}")

    model = load_checkpoint_into_model(args.checkpoint, args.model_name,
                                       args.num_classes, device)

    test_loader = get_testing_dataloader(args.dataset, args.batch_size, model=args.model_name)

    acc, f1 = evaluate(model, test_loader, device, args.num_classes)
    print(f"\nTest Accuracy: {acc*100:.2f}%")
    print(f"Test   F1(Macro): {f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the test set")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .ckpt or .pth file")
    parser.add_argument("--model_name", default="resnet18",
                        help="timm model architecture")
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Output dimension of the classifier head")
    parser.add_argument("--dataset", default="CIFAR100",
                        help="CIFAR100 or ImageNet (case‑insensitive)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for the test loader")
    parser.add_argument("--device", default="cuda",
                        help="cuda or cpu")

    args = parser.parse_args()
    main(args)
