#!/usr/bin/env python3
"""
Simple interactive script to test trained chess evaluation model.

Enter FEN positions and see the model's evaluation.
"""

import sys
from pathlib import Path
import torch
import chess
from chess import Board

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models import ChessCNN, ChessResNet
from src.data import BitboardConverter
from src.utils import (
    load_best_model,
    list_best_models,
    print_header,
    print_section,
    print_info,
    print_success,
    print_error,
    print_warning,
    print_metric_table,
    print_separator,
)


def display_board(fen: str):
    """
    Display chess board from FEN string.
    
    Args:
        fen: FEN notation string
    """
    try:
        board = Board(fen)
        print()
        print_separator()
        print(board)
        print_separator()
        
        # Show game state info
        if board.is_checkmate():
            print_warning("CHECKMATE!")
        elif board.is_check():
            print_warning("CHECK!")
        elif board.is_stalemate():
            print_info("STALEMATE")
        elif board.is_insufficient_material():
            print_info("Insufficient material (draw)")
        
    except Exception as e:
        print_error(f"Error displaying board: {e}")


def evaluate_position(model: torch.nn.Module, fen: str, device: str = 'cpu') -> float:
    """
    Evaluate a chess position using the trained model.
    
    Args:
        model: Trained model
        fen: FEN notation string
        device: Device to run inference on
        
    Returns:
        Evaluation score (from White's perspective)
    """
    try:
        # Convert FEN to bitboard
        bitboard = BitboardConverter.convert_fen(fen, representation='full')
        
        # Convert to tensor
        bitboard_tensor = torch.from_numpy(bitboard).float().unsqueeze(0).to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            output = model(bitboard_tensor)
            evaluation = output.item()
        
        return evaluation
        
    except Exception as e:
        print_error(f"Error evaluating position: {e}")
        return None


def main():
    """Main interactive loop."""
    print_header("Chess Model Evaluator")
    
    # Find best model
    print_info("Looking for trained models...")
    models = list_best_models('best_models')
    
    if not models:
        print_error("No trained models found in best_models/ folder!")
        print_info("Please train a model first with: python train_cnn.py")
        return
    
    # Use most recent model
    best_model_info = models[0]
    model_path = best_model_info['model_path']
    
    print_success(f"Found model: {best_model_info['name']}")
    
    if best_model_info.get('best_val_loss'):
        model_stats = {
            'Best Val Loss': f"{best_model_info['best_val_loss']:.6f}",
            'Parameters': f"{best_model_info.get('parameters', 0):,}" if best_model_info.get('parameters') else 'N/A',
            'Saved At': best_model_info.get('saved_at', 'Unknown')
        }
        print_metric_table(model_stats, title="Model Stats")
    print()
    
    # Load model (auto-detect type from metadata)
    print_info("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Auto-detect model type - load_best_model will handle it
        model, metadata = load_best_model(model_path, model_class=None, device=device)
        print_success(f"Model loaded on {device}")
        
        if metadata:
            model_info = {
                'Type': metadata['model']['type'],
                'Parameters': f"{metadata['model']['parameters']:,}",
                'Training Time': metadata['training']['training_time'],
                'Best Val Loss': f"{metadata['results']['best_val_loss']:.6f}",
            }
            print_metric_table(model_info, title="Model Information")
    except Exception as e:
        print_error(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print()
    print_section("Interactive Evaluation")
    print_info("Enter FEN positions to evaluate (or 'quit' to exit)")
    print_info("Examples:")
    print("  • Starting position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("  • Empty input: (press Enter for starting position)")
    print()
    
    # Interactive loop
    while True:
        try:
            # Get FEN input
            fen_input = input("FEN position (or 'quit'): ").strip()
            
            if fen_input.lower() in ['quit', 'exit', 'q']:
                print()
                print_success("Goodbye!")
                break
            
            # Use starting position if empty
            if not fen_input:
                fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            
            # Validate FEN
            try:
                board = Board(fen_input)
            except ValueError as e:
                print_error(f"Invalid FEN: {e}")
                print()
                continue
            
            # Display board
            display_board(fen_input)
            
            # Evaluate
            print_info("Evaluating position...")
            evaluation = evaluate_position(model, fen_input, device=device)
            
            if evaluation is not None:
                # Format evaluation interpretation
                if abs(evaluation) > 50:
                    interpretation = "Mate/Decisive" if evaluation > 0 else "Mate/Decisive (Black)"
                elif abs(evaluation) > 5:
                    interpretation = "Significant advantage" if evaluation > 0 else "Significant advantage (Black)"
                elif abs(evaluation) > 1:
                    interpretation = "Small advantage" if evaluation > 0 else "Small advantage (Black)"
                else:
                    interpretation = "Roughly equal"
                
                # Display evaluation with table
                eval_info = {
                    'Evaluation': f"{evaluation:+.2f}",
                    'Interpretation': interpretation,
                    'Raw Score': f"{evaluation:.6f}"
                }
                print_metric_table(eval_info, title="Model Evaluation")
            
            print()
            
        except KeyboardInterrupt:
            print()
            print()
            print_success("Goodbye!")
            break
        except Exception as e:
            print_error(f"Error: {e}")
            print()
            continue


if __name__ == '__main__':
    main()

