CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 8,
    "learning_rate": 5e-6,
    "adam_epsilon": 1e-8,
    "num_epochs": 5,
    "beam_size": 5,
    "max_length": 100,
    "pretrained_model_path": "path_to_pretrained_model",
    "model_name": "roberta-base",
    "data_path": "path_to_data",
    "train_filename": "train.pkl",
    "val_filename": "val.pkl",
}
