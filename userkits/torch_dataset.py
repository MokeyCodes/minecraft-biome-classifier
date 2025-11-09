import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
from typing import Tuple, Union, Optional, Dict, List


class MinecraftTorchDataset(Dataset):
    def __init__(self, 
                 data_path: str,
                 mode: str = 'train',
                 return_type: str = 'torch', 
                 transform: Optional[T.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            data_path: Path to the data directory containing biome folders
            return_type: Format to return data in ('torch', 'numpy', 'pil'). Defaults to 'numpy'.
            transform: Optional transform to be applied on a sample (only for torch return_type)
            target_size: Target size to resize images to (height, width)
        """
        self.data_path = data_path
        self.mode = mode.lower()
        self.return_type = return_type.lower()
        self.transform = transform
        self.target_size = target_size
        if self.mode not in { 'train', 'eval' }:
            raise ValueError(f"mode must be 'train' or 'eval', got {self.mode}")
        
        # Get all image paths and their corresponding labels / file_ids
        self.file_paths, self.labels = self.get_image_paths_and_labels()
        
        if self.mode == 'train':
            self.biome_to_class = self.create_biome_mapping()
            self.class_to_biome = {v: k for k, v in self.biome_to_class.items()}
            self.num_classes = len(self.biome_to_class)
        else:
            self.biome_to_class = {}
            self.class_to_biome = {}
            self.num_classes = 0
        
        # Validate return type
        valid_types = ['torch', 'numpy', 'pil']
        if self.return_type not in valid_types:
            raise ValueError(f"return_type must be one of {valid_types}, got {self.return_type}")

    def get_image_paths_and_labels(self) -> Tuple[List[str], List[str]]:
        """
        Get all image paths and extract biome numbers as labels.
        
        Returns:
            Tuple of (image_paths, biome_numbers)
        """
        image_paths: List[str] = []
        labels: List[str] = []
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        if self.mode == 'train':
            for biome_name in os.listdir(self.data_path):
                biome_path = os.path.join(self.data_path, biome_name)
                if not os.path.isdir(biome_path):
                    continue
                for img_name in os.listdir(biome_path):
                    _, ext = os.path.splitext(img_name)
                    if ext.lower() not in exts:
                        continue
                    img_path = os.path.join(biome_path, img_name)
                    image_paths.append(img_path)
                    labels.append(biome_name)
        else:
            for img_name in os.listdir(self.data_path):
                img_path = os.path.join(self.data_path, img_name)
                if os.path.isdir(img_path):
                    continue
                _, ext = os.path.splitext(img_name)
                if ext.lower() not in exts:
                    continue
                image_paths.append(img_path)
                labels.append(img_name)
        
        if len(image_paths) == 0:
            raise ValueError(f"No valid images found in {self.data_path}")
        
        if self.mode == 'train':
            print(f"Found {len(image_paths)} images across {len(set(labels))} biomes")
        else:
            print(f"Found {len(image_paths)} eval images")
        return image_paths, labels

    def create_biome_mapping(self) -> Dict[str, int]:
        """
        Create mapping from biome numbers to sequential class indices.
        
        Returns:
            Dictionary mapping biome_number -> class_index
        """
        unique_biomes = sorted(set(self.labels))
        return {biome_num: idx for idx, biome_num in enumerate(unique_biomes)}

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, str, str], 
                                           Tuple[np.ndarray, str, str], 
                                           Tuple[Image.Image, str, str]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, label) where format depends on return_type
        """
        # Get file path
        file_path = self.file_paths[idx]
        
        if self.mode == 'train':
            label = self.labels[idx]  # biome/class name
        else:
            label = self.labels[idx]  # file_id
        file_id = os.path.basename(file_path)
        
        # Load and process image
        image = Image.open(file_path).convert("RGB")
        
        # Resize image to target size
        image = image.resize(self.target_size, Image.LANCZOS)
        
        if self.return_type == 'pil':
            return image, label, file_id
        
        elif self.return_type == 'numpy':
            # Convert to numpy array (H, W, C) with values in [0, 255]
            image_array = np.array(image, dtype=np.uint8)
            return image_array, label, file_id
        
        elif self.return_type == 'torch':
            # Convert to tensor and apply transforms
            image_tensor = T.ToTensor()(image)  # Converts to (C, H, W) with values in [0, 1]
            
            if self.transform:
                image_tensor = self.transform(image_tensor)
            
            return image_tensor, label, file_id

        
        else:
            raise ValueError(f"Unsupported return_type: {self.return_type}")

    def get_biome_info(self) -> Dict[str, Union[int, Dict[int, int]]]:
        """
        Get information about the biomes in the dataset.
        
        Returns:
            Dictionary with biome statistics and mappings
        """
        if self.mode != 'train':
            raise RuntimeError("get_biome_info is only available in train mode")
        biome_counts = {}
        for biome_num in self.labels:
            biome_counts[biome_num] = biome_counts.get(biome_num, 0) + 1
        
        return {
            'num_classes': self.num_classes,
            'biome_to_class': self.biome_to_class,
            'class_to_biome': self.class_to_biome,
            'biome_counts': biome_counts,
            'total_images': len(self.file_paths)
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Tensor of class weights for use in loss functions
        """
        if self.mode != 'train':
            raise RuntimeError("get_class_weights is only available in train mode")
        biome_counts = {}
        for biome_num in self.labels:
            biome_counts[biome_num] = biome_counts.get(biome_num, 0) + 1
        
        # Convert to class-indexed counts
        class_counts = [biome_counts[self.class_to_biome[i]] for i in range(self.num_classes)]
        
        # Calculate inverse frequency weights
        total_samples = sum(class_counts)
        weights = [total_samples / (self.num_classes * count) for count in class_counts]
        
        return torch.FloatTensor(weights)

    def split_dataset(self, 
                     train_ratio: float = 0.7, 
                     val_ratio: float = 0.15, 
                     test_ratio: float = 0.15,
                     random_seed: int = 42) -> Tuple['MinecraftTorchDataset', 'MinecraftTorchDataset', 'MinecraftTorchDataset']:
        """
        Split the dataset into train/validation/test sets while maintaining class distribution.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set  
            test_ratio: Proportion for test set
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if self.mode != 'train':
            raise RuntimeError("split_dataset is only available in train mode")
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        import random
        random.seed(random_seed)
        
        # Group indices by biome for stratified splitting
        biome_indices = {}
        for idx, biome_num in enumerate(self.labels):
            if biome_num not in biome_indices:
                biome_indices[biome_num] = []
            biome_indices[biome_num].append(idx)
        
        train_indices, val_indices, test_indices = [], [], []
        
        # Split each biome proportionally
        for biome_num, indices in biome_indices.items():
            random.shuffle(indices)
            n_samples = len(indices)
            
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            
            train_indices.extend(indices[:n_train])
            val_indices.extend(indices[n_train:n_train + n_val])
            test_indices.extend(indices[n_train + n_val:])
        
        # Create subset datasets
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices)
        test_dataset = self._create_subset(test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, indices: List[int]) -> 'MinecraftTorchDataset':
        """Create a subset dataset with given indices."""
        subset = MinecraftTorchDataset.__new__(MinecraftTorchDataset)
        subset.data_path = self.data_path
        subset.mode = self.mode
        subset.return_type = self.return_type
        subset.transform = self.transform
        subset.target_size = self.target_size
        subset.biome_to_class = self.biome_to_class
        subset.class_to_biome = self.class_to_biome
        subset.num_classes = self.num_classes
        
        subset.file_paths = [self.file_paths[i] for i in indices]
        subset.labels = [self.labels[i] for i in indices]
        
        return subset