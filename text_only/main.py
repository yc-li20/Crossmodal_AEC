import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from torch.optim import AdamW
import numpy as np
import random
import time
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
from model import Text2TextModel
from utils import parse_args

# Define your custom dataset if necessary
class CustomDataset(Dataset):
    def __init__(self, data_truth, data_trans, tokenizer, source_length, target_length):
        self.data_truth = data_truth
        self.data_trans = data_trans
        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length

    def __len__(self):
        return len(self.data_truth)

    def __getitem__(self, idx):
        source_text = self.data_truth[idx]
        target_text = self.data_trans[idx]
        
        source_tokens = self.tokenizer.encode_plus(
            source_text,
            max_length=self.source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_tokens = self.tokenizer.encode_plus(
            target_text,
            max_length=self.target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'source_ids': source_tokens['input_ids'].squeeze(0),
            'source_mask': source_tokens['attention_mask'].squeeze(0),
            'target_ids': target_tokens['input_ids'].squeeze(0),
            'target_mask': target_tokens['attention_mask'].squeeze(0),
        }

def wer_calculation(output_texts, target_texts):
    total_distance = 0
    total_length = 0

    for output_text, target_text in zip(output_texts, target_texts):
        output_tokens = word_tokenize(output_text.lower())
        target_tokens = word_tokenize(target_text.lower())

        distance = edit_distance(output_tokens, target_tokens)
        total_distance += distance
        total_length += len(target_tokens)

    wer = total_distance / total_length

    return wer

def bleu_gleu_calculation(output_texts, target_texts):
    outputs = []
    targets = []
    
    for output_text, target_text in zip(output_texts, target_texts):
        output_tokens = word_tokenize(output_text.strip().lower())
        target_tokens = [word_tokenize(target_text.strip().lower())]
        
        outputs.append(output_tokens)
        targets.append(target_tokens)
    
    bleu_score = corpus_bleu(targets, outputs)
    gleu_score = corpus_gleu(targets, outputs)
    
    return bleu_score, gleu_score

def main():
    args = parse_args()

    # Initialize tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    config = RobertaConfig.from_pretrained('roberta-base')
    model = Text2TextModel(
        RobertaModel.from_pretrained('roberta-base'),
        TransformerDecoder(config, 1),
        config,
        args.beam_size,
        args.target_length,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id
    )

    # Load datasets
    train_dataset = CustomDataset(
        args.data_truth,
        args.data_trans,
        tokenizer,
        args.source_length,
        args.target_length
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            source_ids = batch['source_ids']
            source_mask = batch['source_mask']
            target_ids = batch['target_ids']
            target_mask = batch['target_mask']

            optimizer.zero_grad()
            loss, tokens, count = model(source_ids, source_mask, target_ids, target_mask, audio)
            total_loss += loss.item()
            total_tokens += tokens.item()

            loss.backward()
            optimizer.step()

        avg_loss = total_loss / total_tokens
        print(f"Epoch {epoch + 1}/{args.num_epochs}, Average Loss: {avg_loss}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            generated_texts = []
            for val_batch in val_loader:
                # Process validation batch
                # Calculate validation loss
                # Collect generated texts

            # Calculate evaluation metrics
            val_truth_revised = []

            for target_text in val_truth:
                encoded = tokenizer(target_text)
                decoded = tokenizer.decode(encoded["input_ids"], clean_up_tokenization_spaces=True, skip_special_tokens=True)
                val_truth_revised.append(decoded)

            generated_texts = []  # replace with actual generated texts from model

            wer = wer_calculation(generated_texts, val_truth_revised)
            bleu, gleu = bleu_gleu_calculation(generated_texts, val_truth_revised)
            print(f"Epoch {epoch + 1}/{args.num_epochs}, WER: {wer:.4f}, BLEU: {bleu:.4f}, GLEU: {gleu:.4f}")

    # Save model if needed
    # torch.save(model.state_dict(), 'model.pth')

if __name__ == "__main__":
    main()
