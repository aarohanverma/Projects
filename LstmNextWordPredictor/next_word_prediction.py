#!/usr/bin/env python
"""
Next Word Prediction using an LSTM model in PyTorch with advanced improvements.
---------------------------------------------------------------------------------
This script supports two modes:

Training Mode (with --train):
  - Loads data from CSV (must contain a 'data' column)
  - Trains a SentencePiece model for subword tokenization (if not already available)
  - Uses SentencePiece to tokenize text and create a Dataset of (input_sequence, target) pairs
  - Builds and trains an LSTM-based model enhanced with:
      * Extra fully connected layer (with ReLU and dropout)
      * Layer Normalization after LSTM outputs
      * Label Smoothing Loss for improved regularization
      * Gradient clipping, Adam optimizer with weight decay, and ReduceLROnPlateau scheduling
  - Saves training/validation loss graphs
  - Converts and saves the model to TorchScript for production deployment

Inference Mode (with --inference "Your sentence"):
  - Loads the saved SentencePiece model and the TorchScript (or checkpoint) model
  - Runs inference to predict the top 3 next words/subwords

Usage:
  Training mode:
      python next_word_prediction.py --data_path data.csv --train
  Inference mode:
      python next_word_prediction.py --inference "How do you"
"""

import os
import sys
import argparse
import logging
import random
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Import SentencePiece
import sentencepiece as spm

# ---------------------- Global Definitions ----------------------
PAD_TOKEN = '<PAD>'  # For padding (id will be 0)
UNK_TOKEN = '<UNK>'
# We use SentencePiece so our tokens come from the trained model

# Set up logging to stdout for Colab compatibility
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------- Label Smoothing Loss ----------------------
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

# ---------------------- SentencePiece Functions ----------------------
def train_sentencepiece(corpus, model_prefix, vocab_size):
    temp_file = "sp_temp.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        for sentence in corpus:
            f.write(sentence.strip() + "\n")
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='unigram'
    )
    os.remove(temp_file)
    logging.info("SentencePiece model trained and saved with prefix '%s'", model_prefix)

def load_sentencepiece_model(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    logging.info("Loaded SentencePiece model from %s", model_path)
    return sp

# ---------------------- Dataset using SentencePiece ----------------------
class NextWordSPDataset(Dataset):
    def __init__(self, sentences, sp):
        logging.info("Initializing NextWordSPDataset with %d sentences", len(sentences))
        self.sp = sp
        self.samples = []
        self.prepare_samples(sentences)
        logging.info("Total samples generated: %d", len(self.samples))
    
    def prepare_samples(self, sentences):
        for idx, sentence in enumerate(sentences):
            token_ids = self.sp.encode(sentence.strip(), out_type=int)
            for i in range(1, len(token_ids)):
                self.samples.append((
                    torch.tensor(token_ids[:i], dtype=torch.long),
                    torch.tensor(token_ids[i], dtype=torch.long)
                ))
            if (idx + 1) % 1000 == 0:
                logging.debug("Processed %d/%d sentences", idx + 1, len(sentences))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def sp_collate_fn(batch):
    inputs, targets = zip(*batch)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    logging.debug("Batch collated: inputs shape %s, targets shape %s", padded_inputs.shape, targets.shape)
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
        # Logging calls removed to allow TorchScript conversion.
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        last_output = output[:, -1, :]
        norm_output = self.layer_norm(last_output)
        norm_output = self.dropout(norm_output)
        fc1_out = torch.relu(self.fc1(norm_output))
        fc1_out = self.dropout(fc1_out)
        logits = self.fc2(fc1_out)
        return logits

# ---------------------- Training and Evaluation ----------------------
def train_model(model, train_loader, valid_loader, optimizer, criterion, scheduler, device,
                num_epochs, patience, model_save_path, clip_value=5):
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    logging.info("Starting training for %d epochs", num_epochs)

    for epoch in range(num_epochs):
        logging.info("Epoch %d started...", epoch + 1)
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 50 == 0:
                logging.debug("Epoch %d, Batch %d: Loss = %.4f", epoch + 1, batch_idx + 1, loss.item())
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        logging.info("Epoch %d training completed. Avg Train Loss: %.4f", epoch + 1, avg_train_loss)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valid_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val_loss += loss.item()
                if (batch_idx + 1) % 50 == 0:
                    logging.debug("Validation Epoch %d, Batch %d: Loss = %.4f", epoch + 1, batch_idx + 1, loss.item())
        avg_val_loss = total_val_loss / len(valid_loader)
        val_losses.append(avg_val_loss)
        logging.info("Epoch %d validation completed. Avg Val Loss: %.4f", epoch + 1, avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info("Checkpoint saved at epoch %d with Val Loss: %.4f", epoch + 1, avg_val_loss)
        else:
            patience_counter += 1
            logging.info("No improvement in validation loss for %d consecutive epoch(s).", patience_counter)
            if patience_counter >= patience:
                logging.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig("loss_graph.png")
    logging.info("Loss graph saved as loss_graph.png")
    
    return train_losses, val_losses

def predict_next_word(model, sentence, sp, device, topk=3):
    """
    Given a partial sentence, uses SentencePiece to tokenize and predicts the top k next words.
    """
    logging.info("Predicting top %d next words for input sentence: '%s'", topk, sentence)
    model.eval()
    token_ids = sp.encode(sentence.strip(), out_type=int)
    logging.debug("Token IDs for prediction: %s", token_ids)
    if len(token_ids) == 0:
        logging.warning("No tokens found in input sentence.")
        return []
    input_seq = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_seq)
        probabilities = torch.softmax(logits, dim=-1)
        topk_result = torch.topk(probabilities, k=topk, dim=-1)
        top_indices = topk_result.indices.squeeze(0).tolist()
    predicted_pieces = [sp.id_to_piece(idx) for idx in top_indices]
    cleaned_predictions = [piece.lstrip("▁") for piece in predicted_pieces]
    logging.info("Predicted top %d next words/subwords: %s", topk, cleaned_predictions)
    return cleaned_predictions

# ---------------------- Main Function ----------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # Inference-only mode
    if args.inference is not None:
        logging.info("Running in inference-only mode with input: '%s'", args.inference)
        if not os.path.exists(args.sp_model_path):
            logging.error("SentencePiece model not found at %s. Cannot run inference.", args.sp_model_path)
            return
        sp = load_sentencepiece_model(args.sp_model_path)
        if os.path.exists(args.scripted_model_path):
            logging.info("Loading TorchScript model from %s", args.scripted_model_path)
            model = torch.jit.load(args.scripted_model_path, map_location=device)
        elif os.path.exists(args.model_save_path):
            logging.info("Loading model checkpoint from %s", args.model_save_path)
            model = LSTMNextWordModel(vocab_size=sp.get_piece_size(),
                                      embed_dim=args.embed_dim,
                                      hidden_dim=args.hidden_dim,
                                      num_layers=args.num_layers,
                                      dropout=args.dropout,
                                      fc_dropout=0.3)
            model.load_state_dict(torch.load(args.model_save_path, map_location=device))
            model.to(device)
        else:
            logging.error("No model checkpoint found. Exiting.")
            return
        predictions = predict_next_word(model, args.inference, sp, device, topk=1)
        logging.info("Input: '%s' -> Predicted next words: %s", args.inference, predictions)
        return

    # Training mode
    logging.info("Loading data from %s...", args.data_path)
    df = pd.read_csv(args.data_path)
    if 'data' not in df.columns:
        logging.error("CSV file must contain a 'data' column. Exiting.")
        return
    sentences = df['data'].tolist()
    logging.info("Total sentences loaded: %d", len(sentences))
    
    if not os.path.exists(args.sp_model_path):
        logging.info("SentencePiece model not found at %s. Training new model...", args.sp_model_path)
        train_sentencepiece(sentences, args.sp_model_prefix, args.vocab_size)
    sp = load_sentencepiece_model(args.sp_model_path)
    
    train_sentences = sentences[:int(len(sentences) * args.train_split)]
    valid_sentences = sentences[int(len(sentences) * args.train_split):]
    train_dataset = NextWordSPDataset(train_sentences, sp)
    valid_dataset = NextWordSPDataset(valid_sentences, sp)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=sp_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=sp_collate_fn)
    logging.info("DataLoaders created: %d training batches, %d validation batches",
                 len(train_loader), len(valid_loader))
    
    vocab_size = sp.get_piece_size()
    model = LSTMNextWordModel(vocab_size=vocab_size,
                              embed_dim=args.embed_dim,
                              hidden_dim=args.hidden_dim,
                              num_layers=args.num_layers,
                              dropout=args.dropout,
                              fc_dropout=0.3)
    model.to(device)
    
    criterion = LabelSmoothingLoss(smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    logging.info("Loss function, optimizer, and scheduler initialized.")
    
    if args.train:
        logging.info("Training mode is ON.")
        if os.path.exists(args.model_save_path):
            logging.info("Existing checkpoint found at %s. Loading weights...", args.model_save_path)
            model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        else:
            logging.info("No checkpoint found. Training from scratch.")
        train_losses, val_losses = train_model(model, train_loader, valid_loader, optimizer, criterion,
                                                scheduler, device, args.num_epochs, args.patience,
                                                args.model_save_path)
        scripted_model = torch.jit.script(model)
        scripted_model.save(args.scripted_model_path)
        logging.info("Model converted to TorchScript and saved to %s", args.scripted_model_path)
    else:
        logging.info("Training flag not set. Skipping training and running inference demo.")
        if not os.path.exists(args.model_save_path):
            logging.error("No model checkpoint found. Exiting.")
            return
    

# ---------------------- Entry Point ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Next Word Prediction using LSTM in PyTorch with SentencePiece and advanced techniques")
    parser.add_argument('--data_path', type=str, default='data.csv', help="Path to CSV file with a 'data' column (required for training)")
    parser.add_argument('--vocab_size', type=int, default=10000, help="Vocabulary size for SentencePiece")
    parser.add_argument('--train_split', type=float, default=0.9, help="Fraction of data to use for training")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training")
    parser.add_argument('--embed_dim', type=int, default=256, help="Dimension of word embeddings")
    parser.add_argument('--hidden_dim', type=int, default=256, help="Hidden dimension for LSTM")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of LSTM layers")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate in LSTM")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="Weight decay (L2 regularization) for optimizer")
    parser.add_argument('--num_epochs', type=int, default=25, help="Number of training epochs")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--label_smoothing', type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help="Path to save the best model checkpoint")
    parser.add_argument('--scripted_model_path', type=str, default='best_model_scripted.pt', help="Path to save the TorchScript model")
    parser.add_argument('--sp_model_prefix', type=str, default='spm', help="Prefix for SentencePiece model files")
    parser.add_argument('--sp_model_path', type=str, default='spm.model', help="Path to load/save the SentencePiece model")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--train', action='store_true', help="Flag to enable training mode. If not set, runs inference/demo using saved checkpoint.")
    parser.add_argument('--inference', type=str, default=None, help="Input sentence for inference-only mode")
    
    args, unknown = parser.parse_known_args()
    logging.info("Arguments parsed: %s", args)
    main(args)
