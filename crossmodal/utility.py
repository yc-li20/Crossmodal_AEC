import torch
import pickle
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu

def wer_calculation(predictions, references):
    wers = [wer(ref.split(), pred.split()) for ref, pred in zip(references, predictions)]
    return sum(wers) / len(wers)

def bleu_gleu_calculation(predictions, references):
    references = [[ref.split()] for ref in references]
    predictions = [pred.split() for pred in predictions]
    bleu = corpus_bleu(references, predictions)
    gleu = corpus_gleu(references, predictions)
    return bleu, gleu

def load_data(config):
    with open(config["data_path"] + config["train_filename"], "rb") as f:
        train_data, train_audio = pickle.load(f)
    with open(config["data_path"] + config["val_filename"], "rb") as f:
        val_data, val_audio = pickle.load(f)
    return train_data, val_data, train_audio, val_audio

def prepare_data(tokenizer, train_data, val_data, config):
    train_source_ids, train_source_mask, train_target_ids, train_target_mask = process_data(tokenizer, train_data, config["max_length"])
    val_source_ids, val_source_mask, val_target_ids, val_target_mask = process_data(tokenizer, val_data, config["max_length"])
    return train_source_ids, train_source_mask, train_target_ids, train_target_mask, val_source_ids, val_source_mask, val_target_ids, val_target_mask

def process_data(tokenizer, data, max_length):
    source_texts, target_texts = data
    source_ids, source_mask, target_ids, target_mask = [], [], [], []

    for source, target in zip(source_texts, target_texts):
        source_encoded = tokenizer(source, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        target_encoded = tokenizer(target, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        source_ids.append(source_encoded["input_ids"])
        source_mask.append(source_encoded["attention_mask"])
        target_ids.append(target_encoded["input_ids"])
        target_mask.append(target_encoded["attention_mask"])

    return torch.cat(source_ids, dim=0), torch.cat(source_mask, dim=0), torch.cat(target_ids, dim=0), torch.cat(target_mask, dim=0)
