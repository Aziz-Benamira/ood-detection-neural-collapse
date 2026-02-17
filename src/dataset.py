"""
utility to load the datasets used in the project.

We train on CIFAR-100 (in-distribution data) and evaluate both
on CIFAR-100 and on some datasets that the model never saw during training
(OOD datasets like SVHN and DTD).

Idea: if the model is well calibrated, it should be less confident
on OOD samples than on CIFAR-100.
"""

import logging
from typing import Dict, Optional, Tuple

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

logger = logging.getLogger(_name_)


def _build_transforms(
    cifar_mean: Tuple[float, ...],
    cifar_std: Tuple[float, ...],
) -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Build the different transform pipelines.

    We use:
    - data augmentation for training
    - simple normalization for testing
    - resizing + normalization for OOD datasets
      (since some of them are not 32x32 like CIFAR)
    """

    # Training: small augmentations to improve generalization
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # small random shifts
        transforms.RandomHorizontalFlip(),      # horizontal symmetry
        transforms.ToTensor(),                  # convert to tensor
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # Testing: no randomness, just normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    # OOD datasets might not be 32x32, so we resize them first
    ood_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std),
    ])

    return train_transform, test_transform, ood_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 2,
    cifar_mean: Tuple[float, ...] = (0.5071, 0.4867, 0.4408),
    cifar_std: Tuple[float, ...] = (0.2675, 0.2565, 0.2761),
) -> Dict[str, Optional[DataLoader]]:
    """
    Create all dataloaders used in the project.

    Returns a dictionary containing:
    - 'train' : CIFAR-100 training loader
    - 'test'  : CIFAR-100 test loader
    - 'svhn'  : SVHN test loader (OOD)
    - 'dtd'   : DTD test loader (OOD, may be None if download fails)
    """

    train_tf, test_tf, ood_tf = _build_transforms(cifar_mean, cifar_std)

 
    # CIFAR-100 (in-distribution)

    train_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf,
    )

    test_dataset = torchvision.datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_tf,
    )

    # SVHN (OOD dataset)

    svhn_dataset = torchvision.datasets.SVHN(
        root=data_dir,
        split='test',
        download=True,
        transform=ood_tf,
    )

    # DTD (OOD dataset)
   
    dtd_dataset = None
    try:
        dtd_dataset = torchvision.datasets.DTD(
            root=data_dir,
            split='test',
            download=True,
            transform=ood_tf,
        )
        logger.info("Loaded DTD with %d images", len(dtd_dataset))
    except Exception as e:
        logger.warning("Could not load DTD (%s). Continuing without it.", e)

    # Build DataLoaders
    
    common_kw = dict(num_workers=num_workers, pin_memory=True)

    loaders: Dict[str, Optional[DataLoader]] = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **common_kw),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **common_kw),
        'svhn': DataLoader(svhn_dataset, batch_size=batch_size, shuffle=False, **common_kw),
        'dtd': (
            DataLoader(dtd_dataset, batch_size=batch_size, shuffle=False, **common_kw)
            if dtd_dataset is not None else None
        ),
    }

    return loaders