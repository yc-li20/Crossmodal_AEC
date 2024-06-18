import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="ASR Error Correction using RoBERTa")

    # Add arguments here as needed
    parser.add_argument("--data_truth", type=str, default="your_groundtruth_path.txt",
                        help="Path to ground truth data")
    parser.add_argument("--data_trans", type=str, default="your_asrtranscript_path.txt",
                        help="Path to ASR transcript data")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=40,
                        help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--model_path", type=str, default="path_to_your_commonvoice_ted_model.bin",
                        help="Path to pretrained model weights")
    parser.add_argument("--train_data_amount", type=int, default=0,
                        help="Amount of training data to use")
    parser.add_argument("--source_length", type=int, default=128,
                        help="Maximum length of source sequences")
    parser.add_argument("--target_length", type=int, default=128,
                        help="Maximum length of target sequences")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for beam search decoding")
    
    return parser.parse_args()
