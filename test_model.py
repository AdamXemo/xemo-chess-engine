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


def find_checkpoints():
    """Find all available checkpoints."""
    from pathlib import Path
    
    checkpoints = []
    
    # Check best_models folder
    best_models = list_best_models('best_models')
    for model_info in best_models:
        checkpoints.append({
            'path': model_info['model_path'],
            'name': model_info['name'],
            'type': 'best_model',
            'val_loss': model_info.get('best_val_loss'),
            'saved_at': model_info.get('saved_at'),
        })
    
    # Check experiments/checkpoints folders
    checkpoint_dir = Path('experiments/checkpoints')
    if checkpoint_dir.exists():
        for exp_dir in checkpoint_dir.iterdir():
            if exp_dir.is_dir():
                # Check for best_model and last_checkpoint
                for checkpoint_name in ['best_model.pth', 'last_checkpoint.pth']:
                    checkpoint_path = exp_dir / checkpoint_name
                    if checkpoint_path.exists():
                        checkpoints.append({
                            'path': str(checkpoint_path),
                            'name': f"{exp_dir.name}/{checkpoint_name.replace('.pth', '')}",
                            'type': 'checkpoint',
                            'val_loss': None,
                            'saved_at': None,
                        })
    
    return checkpoints


def main():
    """Main interactive loop."""
    print_header("Chess Model Evaluator")
    
    # Find all available models/checkpoints
    print_info("Looking for trained models and checkpoints...")
    checkpoints = find_checkpoints()
    
    if not checkpoints:
        print_error("No trained models found!")
        print_info("Please train a model first with: python train_cnn.py or python train_resnet.py")
        return
    
    # Show available checkpoints
    print()
    print_section("Available Models/Checkpoints")
    for i, ckpt in enumerate(checkpoints):
        val_info = f" (val_loss: {ckpt['val_loss']:.6f})" if ckpt['val_loss'] else ""
        model_type = "CNN" if "cnn" in ckpt['name'].lower() else "ResNet" if "resnet" in ckpt['name'].lower() else "?"
        print_info(f"{i+1}. [{model_type}] {ckpt['name']}{val_info}")
    
    # Prefer CNN models (they work correctly)
    cnn_indices = [i for i, ckpt in enumerate(checkpoints) if "cnn" in ckpt['name'].lower()]
    default_idx = cnn_indices[0] if cnn_indices else 0
    
    # Let user choose or use default
    print()
    try:
        choice = input(f"Select model (1-{len(checkpoints)}, Enter for default [{default_idx+1}]): ").strip()
        if choice:
            idx = int(choice) - 1
            if idx < 0 or idx >= len(checkpoints):
                print_warning("Invalid selection, using default")
                idx = default_idx
        else:
            idx = default_idx
    except (ValueError, KeyboardInterrupt):
        idx = default_idx
    
    selected = checkpoints[idx]
    model_path = selected['path']
    
    # Warn about ResNet issues
    if "resnet" in selected['name'].lower():
        print_warning("Note: ResNet models may have evaluation issues. CNN models work correctly.")
    
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
    print_info(f"Loading model: {selected['name']}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Try loading as best_model first (has YAML metadata)
        if model_path.endswith('best_model.pth') or model_path.endswith('_best.pth'):
            model, metadata = load_best_model(model_path, model_class=None, device=device)
        else:
            # Load from checkpoint file
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Try to get model type from checkpoint
            model_type = checkpoint.get('model_type', 'resnet')
            if model_type == 'resnet':
                from src.models.resnet import ChessResNet
                # Reload module to get latest version with clipping fix
                import importlib
                import src.models.resnet
                importlib.reload(src.models.resnet)
                from src.models.resnet import ChessResNet
                
                # Try to infer architecture from state_dict
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                if 'initial_conv.weight' in state_dict:
                    num_filters = state_dict['initial_conv.weight'].shape[0]
                    # Count blocks
                    max_block = -1
                    for key in state_dict.keys():
                        if 'residual_blocks.' in key:
                            try:
                                block_num = int(key.split('residual_blocks.')[1].split('.')[0])
                                max_block = max(max_block, block_num)
                            except:
                                pass
                    num_blocks = max_block + 1 if max_block >= 0 else 15
                    
                    model = ChessResNet(input_channels=23, num_blocks=num_blocks, num_filters=num_filters)
                else:
                    model = ChessResNet()
                model.load_state_dict(state_dict)
                
                # Fix BatchNorm stats if corrupted
                for module in model.modules():
                    if isinstance(module, torch.nn.BatchNorm2d):
                        if torch.abs(module.running_mean).max() > 1000 or module.running_var.max() > 1000000:
                            module.running_mean.zero_()
                            module.running_var.fill_(1.0)
                            module.track_running_stats = False
            else:
                from src.models import ChessCNN
                model = ChessCNN()
                model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            
            model.to(device)
            model.eval()
            metadata = checkpoint.get('config', {})
        
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

