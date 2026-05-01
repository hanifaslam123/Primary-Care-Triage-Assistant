"""
Image preprocessing and augmentation pipelines.

Training transforms include heavy augmentation to improve generalization
on the 5,000+ clinical image dataset.
Inference transforms use only resize + normalize (no randomness).
"""

from torchvision import transforms

# ImageNet mean/std (used because backbone is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """
    Heavy augmentation pipeline for training.

    Includes:
    - Random resized crop (simulates different zoom levels in clinical photos)
    - Horizontal/vertical flips
    - Color jitter (lighting variation between clinical devices)
    - Random rotation (orientation-invariant features)
    - Gaussian blur (simulates out-of-focus images)
    - Normalization using ImageNet statistics
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1
        ),
        transforms.RandomRotation(degrees=20),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Deterministic transforms for validation and inference.
    No randomness — ensures reproducible predictions.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# Alias for inference
get_inference_transforms = get_val_transforms
