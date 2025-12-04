import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import re
import json
import os
import glob
from collections import Counter
from utils import MLPTextGenerator
from huggingface_hub import hf_hub_download # Already imported, perfect!

# --- Caching Functions ---
# These functions work perfectly as-is, since they
# just need a file *path*, which hf_hub_download will provide.
@st.cache_resource
def load_vocabulary(vocab_path):
    """Loads the word_to_ix and ix_to_word mappings."""
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    # Handle json saving int keys as strings
    ix_to_word = {int(k): v for k, v in vocab['ix_to_word'].items()}
    return vocab['word_to_ix'], ix_to_word

@st.cache_resource
def load_model(model_path, device):
    """Loads a pre-trained model checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model_params = checkpoint['hyperparameters']
    model = MLPTextGenerator(**model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, model_params['CONTEXT_SIZE']

# --- NATURAL LANGUAGE Helpers ---

def generate_next_word_natural(model, sentence, word_to_ix, ix_to_word, context_size, device, temperature=1.0):
    model.eval()
    sentence = re.sub('[^a-zA-Z. ]', '', sentence.strip().lower())
    filter_words = [w for w in sentence.split() if w in word_to_ix]
    
    pad_token = '.' if '.' in word_to_ix else list(ix_to_word.values())[0]
    if len(filter_words) < context_size:
        context = [pad_token] * (context_size - len(filter_words)) + filter_words
    else:
        context = filter_words[-context_size:]
        
    context_ids = [word_to_ix[w] for w in context]
    with torch.no_grad():
        logits = model(torch.tensor([context_ids], dtype=torch.long).to(device))
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_idx = torch.multinomial(probs, 1).item()
        
    return ix_to_word.get(str(pred_idx), ix_to_word.get(pred_idx, "[Unknown]")) 

def append_word_natural(s, nxt):
    s = s.strip()
    if not s: 
        return nxt.capitalize()
    if nxt == '.': 
        return s.rstrip() + nxt
    return s + ' ' + nxt.capitalize() if s.endswith('.') else s + ' ' + nxt

# --- STRUCTURED LANGUAGE Helpers ---

TOKEN_REGEX = re.compile(r"""
    (\\[a-zA-Z]+) |              # LaTeX commands
    (::|->|==|!=|<=|>=|&&|\|\||<<|>>|\+=|-=|\*=|/=|%=|&=|\^=|\|=) |  # Multi-char ops
    ([a-zA-Z_][a-zA-Z0-9_]*) |      # Identifiers
    (\d+\.\d*|\.\d+|\d+) |          # Numbers
    (\S)                            # Other non-whitespace chars
""", re.VERBOSE)

def tokenize_code(line): 
    return [m.group(0) for m in TOKEN_REGEX.finditer(line)]

IS_WORDLIKE = re.compile(r'^(\\[a-zA-Z]+|[a-zA-Z_]\w*|\d+\.\d*|\.\d+|\d+)$')
NO_SPACE_BEFORE = {'.', ',', ';', ':', '%', ')', ']', '}', '—'}
NO_SPACE_AFTER = {'(', '[', '{', '$', '$$', '\\[', '\\('}

def append_token_structured(s, tokens, nxt, in_math):
    last = tokens[-1] if tokens else ''
    is_wordlike, is_last = bool(IS_WORDLIKE.match(nxt)), bool(IS_WORDLIKE.match(last))
    open_math, close_math = nxt in {'$', '$$', '\\[', '\\('}, nxt in {'$', '$$', '\\]', '\\)'}

    if open_math and not in_math:
        if s and s[-1] not in ' \t\n' and last not in NO_SPACE_AFTER: s += ' '
        s += nxt; tokens.append(nxt)
        return s, tokens, not in_math if nxt in {'$', '$$'} else True
    if in_math:
        s += nxt; tokens.append(nxt)
        return s, tokens, False if close_math else True
    if close_math and not in_math:
        if s and s[-1] not in ' \t\n' and last not in NO_SPACE_AFTER: s += ' '
        s += nxt; tokens.append(nxt)
        return s, tokens, False
    if last in {'$', '$$', '\\]', '\\)'}:
        s += (' ' if is_wordlike else '') + nxt; tokens.append(nxt)
        return s, tokens, False
    if nxt in {'-', '--', '—'}:
        if s and s[-1] != ' ': s += ' '
        s += nxt; tokens.append(nxt)
        return s, tokens, False
    if is_wordlike and is_last: s += ' ' + nxt
    elif nxt in NO_SPACE_BEFORE or last in NO_SPACE_AFTER: s += nxt
    else: s += (' ' if s and s[-1] not in ' \t\n' else '') + nxt
    tokens.append(nxt)
    return s, tokens, False

def generate_next_word_structured(model, sentence, word_to_ix, ix_to_word, context_size, device, temperature=1):
    model.eval()
    input_words = tokenize_code(sentence.strip())
    filtered_words = [word for word in input_words if word in word_to_ix] 
    
    pad_token = '.' if '.' in word_to_ix else list(ix_to_word.values())[0]
    if len(filtered_words) >= context_size: 
        context_words = filtered_words[-context_size:]
    else: 
        context_words = [pad_token] * (context_size - len(filtered_words)) + filtered_words
        
    context_indices = [word_to_ix[w] for w in context_words]
    with torch.no_grad():
        logits = model(torch.tensor([context_indices], dtype=torch.long).to(device))
        probs = torch.softmax(logits / temperature, dim=-1)
        pred_idx = torch.multinomial(probs, 1).item()
    return ix_to_word.get(str(pred_idx), ix_to_word.get(pred_idx, "[Unknown]"))

# --- Main Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("Neural Language Generator")

# --- MODIFICATION: Define your Hugging Face Repo ID ---
HF_REPO_ID = "BhavayGoyal/Machine_Learning_Assignment_3"

# --- Sidebar Controls ---
st.sidebar.title("Configuration")

# 1. Language Type Selector
lang_type = st.sidebar.radio(
    "Select Language Type",
    ("Natural Language", "Structured Language (Code/LaTeX)")
)

# 2. Set variables based on language type
if lang_type == "Natural Language":
    model_dir = "Models" # This is now a path *inside* the HF repo
    vocab_file = "vocab.json"
    k_label = "Number of words to predict"
    default_text = "The company is doing"
    info_text = "💡 **How this works:** Words not in the model's vocabulary will be ignored when building the context for prediction."
else:
    model_dir = "Models_Structured" # This is now a path *inside* the HF repo
    vocab_file = "vocab.json"
    k_label = "Number of tokens to predict"
    default_text = r"\begin{proof} Let"
    info_text = "💡 **How this works:** Words not in the model's vocabulary will be ignored when building the context for prediction."

# 3. Model Parameter Selection
st.sidebar.subheader("Model Parameters")
cs = st.sidebar.selectbox("Context Size", [5, 10, 15], index=1)
ed = st.sidebar.selectbox("Embedding Dimension", [32, 64], index=1)
hl = st.sidebar.selectbox("Number of Hidden Layers", [1, 2, 3], index=1)
act = st.sidebar.selectbox("Activation Function", ["relu", "tanh"], index=0)

# 4. Generation Parameters
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider(
    "Temperature (Randomness)", 
    min_value=0.1, max_value=2.0, 
    value=0.8 if lang_type == "Natural Language" else 0.6, 
    step=0.1,
    help="Lower = more predictable, Higher = more random/creative."
)
k_words = st.sidebar.number_input(
    k_label, 
    min_value=1, 
    max_value=200, 
    value=20 if lang_type == "Natural Language" else 50
)
seed = st.sidebar.number_input(
    "Random Seed", 
    min_value=1, max_value=99999, value=42,
    help="Set a specific seed for reproducible results."
)

# --- Construct model name ---
act_name = "Relu" if act == "relu" else "Tanh"
selected_model_name = f"CS{cs}_ED{ed}_HL{hl}_{act_name}.pth"

# These paths are relative to the root of your HF repo
# ALWAYS use forward slashes for repo paths, as they are part of a URL
repo_vocab_path = f"{model_dir}/{vocab_file}"
repo_model_path = f"{model_dir}/{selected_model_name}"

# --- Load Model & Vocab ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    # 1. Download and Load vocabulary
    with st.spinner(f"Downloading vocab: {repo_vocab_path}..."):
        # hf_hub_download will download the file and return its *local cached path*
        local_vocab_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_vocab_path
        )
    # Load from the local cached path
    word_to_ix, ix_to_word = load_vocabulary(local_vocab_path)
    
    # 2. Download and Load model
    with st.spinner(f"Downloading model: {repo_model_path}..."):
        local_model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=repo_model_path
        )
    # Load from the local cached path
    model, CONTEXT_SIZE = load_model(local_model_path, device)
    
    # --- If successful, show the rest of the app ---
    # (This logic was previously inside the `if/else` blocks)
    
    st.sidebar.success(f"Loaded '{selected_model_name}'")
    st.sidebar.info(f"**Actual Model Context:** {CONTEXT_SIZE} tokens/words")
    st.sidebar.caption(f"**Device:** {device.type.upper()}")

    # --- Main App Area ---
    st.info(info_text)
    
    input_text = st.text_area("Enter your starting text:", default_text, height=150)

    if st.button("Generate", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some starting text.")
        else:
            # Set seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            with st.spinner(f"Generating {k_words} items..."):
                progress_bar = st.progress(0.0, text="Starting generation...")

                if lang_type == "Natural Language":
                    # --- Natural Language Generation Loop ---
                    generated_text = input_text
                    for i in range(k_words):
                        progress_bar.progress((i + 1) / k_words, text=f"Generating word {i+1}/{k_words}")
                        next_word = generate_next_word_natural(
                            model, generated_text, word_to_ix, ix_to_word,
                            CONTEXT_SIZE, device, temperature
                        )
                        generated_text = append_word_natural(generated_text, next_word)
                    
                    progress_bar.empty()
                    st.subheader("Generated Text")
                    st.markdown(f"_{generated_text}_")
                
                else:
                    # --- Structured Language Generation Loop ---
                    s = input_text
                    tokens = tokenize_code(s)
                    token_counts = Counter(tokens)
                    in_math = False # Basic check
                    if (token_counts['$'] % 2 != 0) or (token_counts['$$'] % 2 != 0): in_math = True
                    elif token_counts['\\['] != token_counts['\\]']: in_math = True
                    elif token_counts['\\('] != token_counts['\\)']: in_math = True

                    for i in range(k_words):
                        progress_bar.progress((i + 1) / k_words, text=f"Generating token {i+1}/{k_words}")
                        next_token = generate_next_word_structured(
                            model, s, word_to_ix, ix_to_word,
                            CONTEXT_SIZE, device, temperature
                        )
                        s, tokens, in_math = append_token_structured(s, tokens, next_token, in_math)
                    
                    progress_bar.empty()
                    st.subheader("Generated Text")
                    st.code(s, language="latex")

except Exception as e:
    # This will catch errors if the file doesn't exist on HF Hub
    # or if the model fails to load.
    st.error(f"Error loading model '{repo_model_path}' from {HF_REPO_ID}.")
    st.error(f"Details: {e}")
    st.warning("Please ensure the selected model parameters exist on the Hugging Face Hub repository.")