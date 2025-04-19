import os
from torchvision import datasets
from curriculum_data import get_c_scores, get_transform, stratified_split

# Perform the split and save it
def main():
    dataset_name = "CIFAR100"
    model_name = "resnet18"
    transform = get_transform(model_name)

    # Load CIFAR100 training set
    dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    # Load difficulty scores
    difficulties = get_c_scores(dataset_name)["scores"]

    # Define output path
    save_path = "split/cifar100_val_split.npz"
    os.makedirs("split", exist_ok=True)

    # Perform and save the split
    stratified_split(
        dataset,
        difficulties,
        val_ratio=0.2,
        random_state=42,
        save_to=save_path
    )
    print(f"Validation split saved to: {save_path}")

if __name__ == "__main__":
    main()
