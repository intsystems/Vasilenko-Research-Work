from pathlib import Path

import torch


def get_synthetic_sample(batch_size=1, S=32, D=64, scale=1.0, seed=None):
    """Generate Gaussian tokens without changing the global RNG when seeded."""
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    return torch.randn(batch_size, S, D, generator=generator) * scale


def load_cifar10_sample(
    batch_size=16,
    data_dir="./data",
    *,
    indices=None,
    download=False,
    return_metadata=False,
):
    """Load explicit test-set images as pixel-row sequences of shape (B, 32, 96).

    If indices are omitted, use the first ``batch_size`` images. No shuffling or
    learned feature extraction is involved. torchvision is only needed here, so
    synthetic experiments can run without a dataset or torchvision installation.
    """
    import torchvision
    import torchvision.transforms as transforms

    indices = list(range(batch_size)) if indices is None else list(indices)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("CIFAR-10 indices must be nonempty and distinct")
    if any(not isinstance(index, int) or not 0 <= index < 10000 for index in indices):
        raise ValueError("CIFAR-10 test indices must be integers in [0, 10000)")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform
    )
    selected = [dataset[index] for index in indices]
    images = torch.stack([image for image, _ in selected])
    batch, channels, height, width = images.shape
    sequences = images.permute(0, 2, 3, 1).reshape(batch, height, width * channels)
    metadata = {
        "dataset": "CIFAR-10",
        "split": "test",
        "data_dir": str(Path(data_dir).resolve()),
        "indices": indices,
        "labels": [int(label) for _, label in selected],
        "torchvision_version": torchvision.__version__,
        "preprocessing": "RGB pixels mapped to [-1, 1]; one image row per token",
        "sequence_shape": list(sequences.shape),
    }
    return (sequences, metadata) if return_metadata else sequences
