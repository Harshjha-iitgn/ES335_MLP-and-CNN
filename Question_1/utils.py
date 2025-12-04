# All the imports
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.optim as optim
from sklearn.manifold import TSNE

# Neural netword model using Torch
class MLPTextGenerator(nn.Module):
    def __init__(self, VOCAB_SIZE : int, EMBEDDING_DIM : int, CONTEXT_SIZE : int, 
                 NUM_HIDDEN_LAYERS : int, HIDDEN_LAYER_DIM : int, ACTIVATION_TYPE : str):
        super().__init__()
        # define all these values
        self.VOCAB_SIZE = VOCAB_SIZE
        self.EMBEDDING_DIM = EMBEDDING_DIM
        self.CONTEXT_SIZE = CONTEXT_SIZE
        self.NUM_HIDDEN_LAYERS = NUM_HIDDEN_LAYERS
        self.HIDDEN_LAYER_DIM = HIDDEN_LAYER_DIM

        # define activation func        
        if ACTIVATION_TYPE == 'tanh':
            self.activation_fn = nn.Tanh()
        else: # relu
            self.activation_fn = nn.ReLU()

        # define embedding layer
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIM)
        layers = []
        layers.append(nn.Linear(CONTEXT_SIZE*EMBEDDING_DIM, HIDDEN_LAYER_DIM)) # Layer 1 (input layer)
        layers.append(self.activation_fn)
        while (NUM_HIDDEN_LAYERS > 1):
            layers.append(nn.Linear(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)) # Layer 2 if req
            layers.append(self.activation_fn)
            NUM_HIDDEN_LAYERS -= 1
        layers.append(nn.Linear(HIDDEN_LAYER_DIM, VOCAB_SIZE)) # OUTPUT Layer # NO softmax to use logits
        self.mlp = nn.Sequential(*layers)

    def forward(self, X: torch.tensor):
        embeds = self.embedding(X)
        input_layer = embeds.view(embeds.size(0), -1) # Flat all the embeddings
        logits = self.mlp(input_layer)
        return logits
    
def train_model(model, train_loader, val_loader, learning_rate, epochs, device, patience=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}; best_val_loss = float('inf'); patience_counter = 0
    print(f"Starting training on {device} for up to {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        total_train_loss = 0.0
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            # --- Progress print every 50 batches ---
            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} processed. ({len(train_loader) - (batch_idx+1)} left)")

        # Validation
        model.eval()
        total_val_loss, correct_preds, total_preds = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct_preds += (predicted == batch_y).sum().item(); total_preds += batch_y.size(0)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            print(f"No improvement in validation loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                model.load_state_dict(best_model_state)
                break

    model.load_state_dict(best_model_state) # Load the best model
    return history, model
    
def plot_loss(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title('Training vs Validation Loss', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def visualize_embeddings(model, word_to_ix, words_to_plot, SEED):
    embeddings = model.embedding.weight.data.cpu().numpy()
    words = [w for w in words_to_plot if w in word_to_ix]
    if not words:
        print("Error: None of the selected words are in the vocabulary.")
        return
    indices = [word_to_ix[w] for w in words]
    tsne = TSNE(n_components=2, random_state=SEED,
                perplexity=min(30.0, len(words)-1), max_iter=1000, init='pca')
    Y = tsne.fit_transform(embeddings[indices])
    plt.figure(figsize=(15, 10))
    plt.scatter(Y[:, 0], Y[:, 1], s=10)
    for i, w in enumerate(words):
        plt.annotate(w, (Y[i, 0], Y[i, 1]), fontsize=9)
    plt.title('t-SNE Visualization of Word Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)

def generate_next_word_natural(model, sentence, word_to_ix, ix_to_word, context_size, device, temperature=1):
    model.eval()
    sentence = re.sub('[^a-zA-Z ]', '', sentence.strip().lower())
    filter_words = [w for w in sentence.split() if w in word_to_ix]
    if len(filter_words) < context_size: context = ['.'] * (context_size - len(filter_words)) + filter_words
    else: context = filter_words[-context_size:]
    context_ids = [word_to_ix[w] for w in context]
    with torch.no_grad():
        logits = model(torch.tensor([context_ids], dtype=torch.long).to(device))
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_idx = torch.multinomial(probs, 1).item()
    return ix_to_word.get(pred_idx, "[Unknown]")

TOKEN_REGEX = re.compile(r"""
    (\\[a-zA-Z]+) |                       # LaTeX commands
    (::|->|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\^=|\|=) |  # Multi-char ops
    ([a-zA-Z_][a-zA-Z0-9_]*) |            # Identifiers
    (\d+\.\d*|\.\d+|\d+) |                # Numbers
    (\S)                                  # Other non-whitespace chars
""", re.VERBOSE)
def tokenize_code(line): return [m.group(0) for m in TOKEN_REGEX.finditer(line)]
def generate_next_word_structured(model, sentence, word_to_ix, ix_to_word, context_size, device, temperature=1):
    model.eval()
    input_words = tokenize_code(sentence.strip())
    filtered_words = [word for word in input_words if word in word_to_ix] 
    if len(filtered_words) >= context_size: context_words = filtered_words[-context_size:]
    else: context_words = ['.'] * (context_size - len(filtered_words)) + filtered_words
    context_indices = [word_to_ix[w] for w in context_words]
    with torch.no_grad():
        logits = model(torch.tensor([context_indices], dtype=torch.long).to(device))
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_idx = torch.multinomial(probs, 1).item()
    return ix_to_word.get(pred_idx, "[Unknown]")