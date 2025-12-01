"""
Bitboard conversion utilities for chess positions.

Converts FEN notation to bitboard representations for neural network input.
"""

from typing import Tuple
import numpy as np
import chess


# Mapping from piece symbols to channel indices
PIECE_TO_CHANNEL = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}


class BitboardConverter:
    """
    Converts chess positions to bitboard representations.
    
    Supports: 'simple' (13ch), 'essential' (19ch), 'full' (23ch)
    """
    
    REPRESENTATION_CHANNELS = {
        'simple': 13,
        'essential': 19,
        'full': 23
    }
    
    @classmethod
    def convert(cls, board: chess.Board, representation: str = 'full') -> np.ndarray:
        """
        Convert a chess board to bitboard representation.
        
        Args:
            board: python-chess Board object
            representation: Type of representation ('simple', 'essential', or 'full')
            
        Returns:
            numpy array of shape (C, 8, 8) where C depends on representation type
            
        Raises:
            ValueError: If representation type is not recognized
        """
        if representation == 'simple':
            return cls._convert_simple(board)
        elif representation == 'essential':
            return cls._convert_essential(board)
        elif representation == 'full':
            return cls._convert_full(board)
        else:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Must be one of: {list(cls.REPRESENTATION_CHANNELS.keys())}"
            )
    
    @classmethod
    def convert_fen(cls, fen: str, representation: str = 'full') -> np.ndarray:
        """
        Convert a FEN string directly to bitboard representation.
        
        Args:
            fen: FEN notation string
            representation: Type of representation ('simple', 'essential', or 'full')
            
        Returns:
            numpy array of shape (C, 8, 8) where C depends on representation type
        """
        board = chess.Board(fen)
        return cls.convert(board, representation)
    
    @staticmethod
    def _convert_simple(board: chess.Board) -> np.ndarray:
        """
        Convert to simple 13-channel representation.
        
        Channels:
        0-11: Piece positions (one channel per piece type)
        12: Turn indicator (1 = White to move, 0 = Black to move)
        
        Args:
            board: python-chess Board object
            
        Returns:
            numpy array of shape (13, 8, 8)
        """
        bitboard = np.zeros((13, 8, 8), dtype=np.uint8)
        
        # Piece positions (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                channel = PIECE_TO_CHANNEL[piece.symbol()]
                bitboard[channel, row, col] = 1
        
        # Turn indicator (channel 12)
        bitboard[12, :, :] = 1 if board.turn == chess.WHITE else 0
        
        return bitboard
    
    @staticmethod
    def _convert_essential(board: chess.Board) -> np.ndarray:
        """
        Convert to essential 19-channel representation.
        
        Channels:
        0-11: Piece positions
        12: Turn indicator
        13: En passant target square
        14-17: Castling rights (White kingside, White queenside, Black kingside, Black queenside)
        18: Check indicator
        
        Args:
            board: python-chess Board object
            
        Returns:
            numpy array of shape (19, 8, 8)
        """
        bitboard = np.zeros((19, 8, 8), dtype=np.uint8)
        channel_idx = 0
        
        # Channels 0-11: Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                channel = PIECE_TO_CHANNEL[piece.symbol()]
                bitboard[channel, row, col] = 1
        
        channel_idx = 12
        
        # Channel 12: Turn indicator
        bitboard[channel_idx, :, :] = 1 if board.turn == chess.WHITE else 0
        channel_idx += 1
        
        # Channel 13: En passant target square
        if board.ep_square is not None:
            row, col = 7 - (board.ep_square // 8), board.ep_square % 8
            bitboard[channel_idx, row, col] = 1
        channel_idx += 1
        
        # Channels 14-17: Castling rights (4 separate channels)
        if board.has_kingside_castling_rights(chess.WHITE):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.WHITE):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_kingside_castling_rights(chess.BLACK):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.BLACK):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        # Channel 18: Check indicator
        if board.is_check():
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        return bitboard
    
    @staticmethod
    def _convert_full(board: chess.Board) -> np.ndarray:
        """
        Convert to full 23-channel representation with advanced features.
        
        Channels:
        0-11: Piece positions
        12: Turn indicator
        13: En passant target square
        14-17: Castling rights
        18-19: Attack maps (White attacks, Black attacks)
        20-21: Mobility maps (White mobility, Black mobility)
        22: Check indicator
        
        Args:
            board: python-chess Board object
            
        Returns:
            numpy array of shape (23, 8, 8)
        """
        bitboard = np.zeros((23, 8, 8), dtype=np.uint8)
        channel_idx = 0
        
        # Channels 0-11: Piece positions
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - (square // 8), square % 8
                channel = PIECE_TO_CHANNEL[piece.symbol()]
                bitboard[channel, row, col] = 1
        
        channel_idx = 12
        
        # Channel 12: Turn indicator
        bitboard[channel_idx, :, :] = 1 if board.turn == chess.WHITE else 0
        channel_idx += 1
        
        # Channel 13: En passant target square
        if board.ep_square is not None:
            row, col = 7 - (board.ep_square // 8), board.ep_square % 8
            bitboard[channel_idx, row, col] = 1
        channel_idx += 1
        
        # Channels 14-17: Castling rights (4 separate channels)
        if board.has_kingside_castling_rights(chess.WHITE):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.WHITE):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_kingside_castling_rights(chess.BLACK):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        if board.has_queenside_castling_rights(chess.BLACK):
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        # Channels 18-19: Attack maps
        white_attacks, black_attacks = BitboardConverter._compute_attack_maps(board)
        bitboard[channel_idx, :, :] = white_attacks
        channel_idx += 1
        bitboard[channel_idx, :, :] = black_attacks
        channel_idx += 1
        
        # Channels 20-21: Mobility maps
        white_mobility, black_mobility = BitboardConverter._compute_mobility_maps(board)
        bitboard[channel_idx, :, :] = white_mobility
        channel_idx += 1
        bitboard[channel_idx, :, :] = black_mobility
        channel_idx += 1
        
        # Channel 22: Check indicator
        if board.is_check():
            bitboard[channel_idx, :, :] = 1
        channel_idx += 1
        
        return bitboard
    
    @staticmethod
    def _compute_attack_maps(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary attack maps for both sides.
        
        Each square is 1 if it's attacked by that side, 0 otherwise.
        
        Args:
            board: python-chess Board object
            
        Returns:
            Tuple of (white_attacks, black_attacks), each of shape (8, 8)
        """
        white_attacks = np.zeros((8, 8), dtype=np.uint8)
        black_attacks = np.zeros((8, 8), dtype=np.uint8)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                for attacked_square in board.attacks(square):
                    row, col = 7 - (attacked_square // 8), attacked_square % 8
                    if piece.color == chess.WHITE:
                        white_attacks[row, col] = 1
                    else:
                        black_attacks[row, col] = 1
        
        return white_attacks, black_attacks
    
    @staticmethod
    def _compute_mobility_maps(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary mobility maps for both sides.
        
        Each square is 1 if a piece of that side can move from it, 0 otherwise.
        
        Args:
            board: python-chess Board object
            
        Returns:
            Tuple of (white_mobility, black_mobility), each of shape (8, 8)
        """
        white_mobility = np.zeros((8, 8), dtype=np.uint8)
        black_mobility = np.zeros((8, 8), dtype=np.uint8)
        
        for move in board.legal_moves:
            row, col = 7 - (move.from_square // 8), move.from_square % 8
            if board.color_at(move.from_square) == chess.WHITE:
                white_mobility[row, col] = 1
            else:
                black_mobility[row, col] = 1
        
        return white_mobility, black_mobility

