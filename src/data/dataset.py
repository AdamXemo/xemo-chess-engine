"""
PyTorch Dataset for chess position evaluation.

Loads FEN positions and evaluations for training.
"""

from typing import Tuple, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset
import chess

from .bitboard import BitboardConverter


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess position evaluation.
    
    Loads positions from FEN strings and their corresponding evaluations,
    converting them to bitboard representations for neural network input.
    """
    
    def __init__(
        self,
        fen_list: List[str],
        eval_list: List[float],
        representation: str = 'full',
        transform: Optional[callable] = None,
        normalize_eval: bool = False
    ):
        """
        Initialize the chess dataset.
        
        Args:
            fen_list: List of FEN notation strings
            eval_list: List of evaluation scores corresponding to FEN positions
            representation: Bitboard representation type ('simple', 'essential', 'full')
            transform: Optional transform to apply to the bitboard
            normalize_eval: Whether to normalize evaluations to [-1, 1] range
        """
        assert len(fen_list) == len(eval_list), \
            f"FEN list ({len(fen_list)}) and eval list ({len(eval_list)}) must have same length"
        
        self.fen_list = fen_list
        self.eval_list = eval_list
        self.representation = representation
        self.transform = transform
        self.normalize_eval = normalize_eval
        
        # Validate representation type
        if representation not in BitboardConverter.REPRESENTATION_CHANNELS:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Must be one of: {list(BitboardConverter.REPRESENTATION_CHANNELS.keys())}"
            )
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.fen_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (bitboard, evaluation) as PyTorch tensors
            - bitboard: shape (C, 8, 8) where C is number of channels
            - evaluation: shape (1,) - scalar evaluation score
        """
        fen = self.fen_list[idx]
        evaluation = self.eval_list[idx]
        
        # Convert FEN to bitboard
        try:
            board = chess.Board(fen)
            bitboard = BitboardConverter.convert(board, self.representation)
        except Exception as e:
            raise ValueError(f"Error converting FEN at index {idx}: {fen}") from e
        
        # Apply optional transform
        if self.transform is not None:
            bitboard = self.transform(bitboard)
        
        # Normalize evaluation if requested
        if self.normalize_eval:
            # Normalize from [-100, 100] to [-1, 1]
            evaluation = evaluation / 100.0
        
        # Convert to PyTorch tensors
        bitboard_tensor = torch.from_numpy(bitboard).float()
        eval_tensor = torch.tensor([evaluation], dtype=torch.float32)
        
        return bitboard_tensor, eval_tensor
    
    def get_input_shape(self) -> Tuple[int, int, int]:
        """
        Get the shape of the input bitboard.
        
        Returns:
            Tuple of (channels, height, width)
        """
        num_channels = BitboardConverter.REPRESENTATION_CHANNELS[self.representation]
        return (num_channels, 8, 8)


class ChessDataLoader:
    """
    Loads chess data from CSV and creates train/val/test datasets.
    """
    
    @staticmethod
    def load_from_csv(
        csv_path: str,
        max_samples: Optional[int] = None,
        skip_header: bool = True
    ) -> Tuple[List[str], List[float]]:
        """
        Load chess positions and evaluations from CSV file.
        
        CSV format expected: FEN,Evaluation
        Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1,0.0
        
        Args:
            csv_path: Path to the CSV file
            max_samples: Maximum number of samples to load (None = load all)
            skip_header: Whether to skip the first line (header row)
            
        Returns:
            Tuple of (fen_list, eval_list)
        """
        fen_list = []
        eval_list = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            if skip_header:
                next(f)  # Skip header line
            
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Split by comma (last occurrence to handle FEN with commas)
                    # FEN format doesn't actually contain commas, so simple split works
                    parts = line.rsplit(',', 1)
                    if len(parts) != 2:
                        print(f"Warning: Skipping malformed line {i}: {line}")
                        continue
                    
                    fen, eval_str = parts
                    evaluation = float(eval_str)
                    
                    # Validate FEN by trying to parse it
                    chess.Board(fen)
                    
                    fen_list.append(fen)
                    eval_list.append(evaluation)
                    
                except ValueError as e:
                    print(f"Warning: Skipping invalid entry at line {i}: {e}")
                    continue
        
        print(f"Loaded {len(fen_list)} valid positions from {csv_path}")
        return fen_list, eval_list
    
    @staticmethod
    def split_data(
        fen_list: List[str],
        eval_list: List[float],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
        random_seed: Optional[int] = 42
    ) -> Tuple[Tuple[List[str], List[float]], 
               Tuple[List[str], List[float]], 
               Tuple[List[str], List[float]]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            fen_list: List of FEN strings
            eval_list: List of evaluation scores
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of ((train_fens, train_evals), (val_fens, val_evals), (test_fens, test_evals))
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Train, val, and test ratios must sum to 1.0"
        
        n = len(fen_list)
        indices = np.arange(n)
        
        if shuffle:
            if random_seed is not None:
                np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Split data
        train_fens = [fen_list[i] for i in train_indices]
        train_evals = [eval_list[i] for i in train_indices]
        
        val_fens = [fen_list[i] for i in val_indices]
        val_evals = [eval_list[i] for i in val_indices]
        
        test_fens = [fen_list[i] for i in test_indices]
        test_evals = [eval_list[i] for i in test_indices]
        
        print(f"Data split: Train={len(train_fens)}, Val={len(val_fens)}, Test={len(test_fens)}")
        
        return (train_fens, train_evals), (val_fens, val_evals), (test_fens, test_evals)
    
    @staticmethod
    def create_datasets(
        csv_path: str,
        representation: str = 'full',
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        max_samples: Optional[int] = None,
        normalize_eval: bool = False,
        shuffle: bool = True,
        random_seed: Optional[int] = 42
    ) -> Tuple[ChessDataset, ChessDataset, ChessDataset]:
        """
        Convenience method to load data from CSV and create train/val/test datasets.
        
        Args:
            csv_path: Path to the CSV file
            representation: Bitboard representation type
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            max_samples: Maximum number of samples to load
            normalize_eval: Whether to normalize evaluations
            shuffle: Whether to shuffle data before splitting
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load data from CSV
        fen_list, eval_list = ChessDataLoader.load_from_csv(csv_path, max_samples)
        
        # Split data
        (train_fens, train_evals), (val_fens, val_evals), (test_fens, test_evals) = \
            ChessDataLoader.split_data(
                fen_list, eval_list, train_ratio, val_ratio, test_ratio, shuffle, random_seed
            )
        
        # Create datasets
        train_dataset = ChessDataset(
            train_fens, train_evals, representation, normalize_eval=normalize_eval
        )
        val_dataset = ChessDataset(
            val_fens, val_evals, representation, normalize_eval=normalize_eval
        )
        test_dataset = ChessDataset(
            test_fens, test_evals, representation, normalize_eval=normalize_eval
        )
        
        return train_dataset, val_dataset, test_dataset

