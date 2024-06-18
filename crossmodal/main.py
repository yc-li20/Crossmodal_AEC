import os
import time
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from nltk.tokenize import word_tokenize
from torch.nn.parallel import DataParallel
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

from model import Text2TextModel, Beam
from config import CONFIG
from utils import wer_calculation, bleu_gleu_calculation, load_data, prepare_data

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

def main():
    # Load tokenizer and model configuration
    tokenizer = RobertaTokenizer.from_pretrained(CONFIG["model_name"], do_lower_case=True)
    config = RobertaConfig.from_pretrained(CONFIG["model_name"])

    # Load data
    train_data, val_data, train_audio, val_audio = load_data(CONFIG)

    # Prepare data
    train_source_ids, train_source_mask, train_target_ids, train_target_mask, val_source_ids, val_source_mask, val_target_ids, val_target_mask = prepare_data(tokenizer, train_data, val_data, CONFIG)

    # Initialize the model
    encoder = RobertaModel.from_pretrained(CONFIG["model_name"])
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Text2TextModel(encoder, decoder, config, CONFIG["beam_size"], CONFIG["max_length"], tokenizer.cls_token_id, tokenizer.sep_token_id)
    model.load_state_dict(torch.load(CONFIG["pretrained_model_path"]), strict=False)
    model.to(CONFIG["device"])

    # Optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=CONFIG["learning_rate"], eps=CONFIG["adam_epsilon"])

    # Training loop
    train_model(model, tokenizer, optimizer, train_source_ids, train_source_mask, train_target_ids, train_target_mask, val_source_ids, val_source_mask, train_audio, val_audio, val_data, CONFIG)

def train_model(model, tokenizer, optimizer, train_source_ids, train_source_mask, train_target_ids, train_target_mask, val_source_ids, val_source_mask, train_audio, val_audio, val_data, CONFIG):
    wers = []
    bleus = []
    gleus = []

    for epoch in range(CONFIG["num_epochs"]):
        start_time = time.time()
        model.train()
        total_loss = 0

        for i in range(0, len(train_source_ids), CONFIG["batch_size"]):
            batch_source_ids = train_source_ids[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_source_mask = train_source_mask[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_target_ids = train_target_ids[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_target_mask = train_target_mask[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_audio = train_audio[i:i+CONFIG["batch_size"]].to(CONFIG["device"])

            loss, _, _ = model(batch_source_ids, batch_source_mask, batch_target_ids, batch_target_mask, batch_audio)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            total_loss += loss.sum().item()

        avg_train_loss = total_loss / (len(train_source_ids) // CONFIG["batch_size"])

        # Evaluation
        model.eval()
        generated_texts = []

        for i in range(0, len(val_source_ids), CONFIG["batch_size"]):
            batch_source_ids = val_source_ids[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_source_mask = val_source_mask[i:i+CONFIG["batch_size"]].to(CONFIG["device"])
            batch_audio = val_audio[i:i+CONFIG["batch_size"]].to(CONFIG["device"])

            with torch.no_grad():
                preds = model(batch_source_ids, batch_source_mask, None, None, batch_audio)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=True, skip_special_tokens=True)
                    text = text.strip()
                    generated_texts.append(text)

        val_truth_revised = [tokenizer.decode(tokenizer(target)["input_ids"], clean_up_tokenization_spaces=True, skip_special_tokens=True) for target in val_data[0]]
        wer = wer_calculation(generated_texts, val_truth_revised)
        bleu, gleu = bleu_gleu_calculation(generated_texts, val_truth_revised)
        wers.append([epoch+1, wer])
        bleus.append([epoch+1, bleu])
        gleus.append([epoch+1, gleu])

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}: Train Loss = {avg_train_loss:.4f}, WER: {wer:.4f}, BLEU: {bleu:.4f}, GLEU: {gleu:.4f}, Time: {epoch_time:.2f}s")

        torch.cuda.empty_cache()

    best_wer = min(wers, key=lambda x: x[1])
    best_bleu = max(bleus, key=lambda x: x[1])
    best_gleu = max(gleus, key=lambda x: x[1])

    print(f"Best WER: {best_wer}")
    print(f"Best BLEU: {best_bleu}")
    print(f"Best GLEU: {best_gleu}")

if __name__ == "__main__":
    main()
