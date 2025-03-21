#!/usr/bin/env python
"""
Evaluation script for Next Word Prediction model.
Loads the trained model and SentencePiece model,
prepares the validation dataset, and computes:
    - Perplexity (using average loss)
    - Top-k Accuracy (e.g., top-3 accuracy)
Usage:
    python evaluate_next_word.py --data_path data.csv \
         --sp_model_path spm.model --model_save_path best_model.pth \
         [--batch_size 512] [--top_k 3]
"""

import os
import sys
import math
import argparse
import logging
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import sentencepiece as spm

# ---------------------- Logging Configuration ----------------------
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------- Dataset Definition ----------------------
class NextWordSPDataset(Dataset):
    def __init__(self, sentences, sp):
        self.sp = sp
        self.samples = []
        self.prepare_samples(sentences)
    
    def prepare_samples(self, sentences):
        for sentence in sentences:
            token_ids = self.sp.encode(sentence.strip(), out_type=int)
            # For each sentence, create (input_sequence, target) pairs.
            for i in range(1, len(token_ids)):
                self.samples.append((
                    torch.tensor(token_ids[:i], dtype=torch.long),
                    torch.tensor(token_ids[i], dtype=torch.long)
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def sp_collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded_inputs, targets

# ---------------------- Model Definition ----------------------
class LSTMNextWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, fc_dropout=0.3):
        super(LSTMNextWordModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        last_output = output[:, -1, :]
        norm_output = self.layer_norm(last_output)
        norm_output = self.dropout(norm_output)
        fc1_out = torch.relu(self.fc1(norm_output))
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)
        return logits

# ---------------------- Evaluation Functions ----------------------
def evaluate_perplexity(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
    avg_loss = total_loss / total_samples
    perplexity = math.exp(avg_loss)
    return perplexity

def evaluate_topk_accuracy(model, dataloader, k, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            # Get top-k predictions for each sample
            _, topk_indices = torch.topk(logits, k, dim=-1)
            for i in range(len(targets)):
                if targets[i] in topk_indices[i]:
                    correct += 1
            total += targets.size(0)
    accuracy = correct / total if total > 0 else 0
    return accuracy

# ---------------------- Main Evaluation Routine ----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Load SentencePiece model
    if not os.path.exists(args.sp_model_path):
        logging.error("SentencePiece model not found at %s", args.sp_model_path)
        sys.exit(1)
    sp = spm.SentencePieceProcessor()
    sp.load(args.sp_model_path)
    logging.info("Loaded SentencePiece model from %s", args.sp_model_path)
    
    # Load data and prepare validation set
    if not os.path.exists(args.data_path):
        logging.error("Data CSV file not found at %s", args.data_path)
        sys.exit(1)
    df = pd.read_csv(args.data_path)
    if 'data' not in df.columns:
        logging.error("CSV file must contain a 'data' column.")
        sys.exit(1)
    sentences = df['data'].tolist()
    # Use a portion for validation. Here, we assume last 10% is validation.
    split_index = int(len(sentences) * 0.9)
    valid_sentences = sentences[split_index:]
    logging.info("Validation sentences: %d", len(valid_sentences))
    
    valid_dataset = NextWordSPDataset(valid_sentences, sp)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                              shuffle=False, collate_fn=sp_collate_fn)
    
    # Initialize model. You may need to adjust these parameters to match your training.
    vocab_size = sp.get_piece_size()
    embed_dim = args.embed_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    dropout = args.dropout
    model = LSTMNextWordModel(vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    model.to(device)
    
    # Load the trained model weights
    if not os.path.exists(args.model_save_path):
        logging.error("Model checkpoint not found at %s", args.model_save_path)
        sys.exit(1)
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    logging.info("Loaded model checkpoint from %s", args.model_save_path)
    
    # Define the loss criterion.
    # Note: If you used label smoothing during training, you can reuse that here.
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, smoothing=0.1):
            super(LabelSmoothingLoss, self).__init__()
            self.smoothing = smoothing

        def forward(self, pred, target):
            confidence = 1.0 - self.smoothing
            vocab_size = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
            smoothed_target = one_hot * confidence + self.smoothing / (vocab_size - 1)
            log_prob = torch.log_softmax(pred, dim=-1)
            loss = -(smoothed_target * log_prob).sum(dim=1).mean()
            return loss

    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    
    # Evaluate perplexity and top-k accuracy
    val_perplexity = evaluate_perplexity(model, valid_loader, criterion, device)
    topk_accuracy = evaluate_topk_accuracy(model, valid_loader, args.top_k, device)
    
    logging.info("Validation Perplexity: %.4f", val_perplexity)
    logging.info("Top-%d Accuracy: %.4f", args.top_k, topk_accuracy)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Next Word Prediction Model")
    parser.add_argument('--data_path', type=str, default='data.csv', help="Path to CSV file with a 'data' column")
    parser.add_argument('--sp_model_path', type=str, default='spm.model', help="Path to the SentencePiece model file")
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help="Path to the trained model checkpoint")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for evaluation")
    parser.add_argument('--top_k', type=int, default=3, help="Top-k value for computing accuracy")
    # Model hyperparameters (should match those used in training)
    parser.add_argument('--embed_dim', type=int, default=256, help="Embedding dimension")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor")
    
    args = parser.parse_args()
    main(args)

