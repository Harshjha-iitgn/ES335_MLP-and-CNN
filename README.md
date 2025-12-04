# ES335 Assignment – Summary

## **Question 1 — Next-Word Prediction using MLP (5 marks)**

### **1.1 Preprocessing & Vocabulary Construction (0.5 marks)**
- Use two datasets:  
  - **Category I (Natural Language)** – e.g., Paul Graham essays, Wikipedia, Shakespeare, etc.  
  - **Category II (Structured Text)** – e.g., Python/C++ code, IITGN advisory, math textbook.
- Preprocessing:
  - Remove special characters (except period `.`) for natural text.
  - For structured text (e.g., code), treat each line as a statement.
  - Convert to lowercase.
  - Create vocabulary from unique words.
- Report:
  - Vocabulary size  
  - 10 most frequent words  
  - 10 least frequent words  
- Create training pairs `(X, y)` using sliding-window context similar to next-character prediction.

---

### **1.2 Model Design & Training (1 mark)**
- MLP architecture:
  - Embedding: **32 or 64**
  - Hidden layers: **1–2 layers with 1024 neurons**
  - Activation: **ReLU or Tanh**
  - Output: **Softmax over vocabulary**
- Training:
  - Use **500–1000 epochs**
  - Use **train/validation split**
- Report:
  - Training vs validation loss plot  
  - Final validation loss / accuracy  
  - Example predictions + interpretation  

---

### **1.3 Embedding Visualization (1 mark)**
- Use **t-SNE** for >2D embeddings; scatter plot if 2D.
- Select representative words: synonyms, antonyms, verbs, pronouns, random unrelated words.
- Discuss:
  - Clustering behavior  
  - Semantic relationships learned  
  - Differences between datasets  

---

### **1.4 Streamlit Application (1.5 marks)**
- Build a Streamlit app that:
  - Takes input text from user
  - Predicts next **k words or lines**
  - Allows control of:
    - Context length
    - Embedding dimension
    - Activation function
    - Random seed  
    - Temperature (for sampling)
- Handle out-of-vocabulary user words gracefully.
- Provide 2–3 trained model variants to choose from.
- Include the **Streamlit link** at the top of the notebook.

---

### **1.5 Comparative Analysis (1 mark)**
Compare **Category I vs Category II** models:
- Dataset size  
- Vocabulary size  
- Predictability of context  
- Model performance (loss, accuracy)  
- Example generations  
- Embedding visualizations  
- Insights:
  - Natural language vs structured text learnability  
  - Differences in embedding structure  

---

---

## **Question 2 — Moons Dataset & Regularization (3 marks)**

### **Dataset Construction**
- Generate **make-moons manually** (no sklearn), default noise **0.2**.
- Create two additional test sets with noise **0.1** and **0.3**.
- Train/test = **500/500 points**.
- Standardize using **train statistics only**.
- Validation split = **20%** of training set.
- Random seed = **1337**.

---

### **Models to Train**
1. **MLP with Early Stopping**  
   - Patience = 50  
2. **MLP with L1 Regularization**  
   - λ ∈ {1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4}  
   - Report layer-wise sparsity  
   - Plot validation AUROC vs λ  
3. **MLP with L2 Regularization**  
   - Tune penalty coefficient on validation set  
4. **Logistic Regression with Polynomial Features**  
   - Include x₁x₂, x₁², x₂², etc.

---

### **Evaluation Requirements**
- Report **test accuracy** for noise = 0.20  
- Report **robustness accuracy** for noise = 0.10 & 0.30  
- Create table:
  - Test accuracies for all 4 models on 3 noise levels  
  - Parameter count of each model  
- Plot **decision boundaries** for all 4 models (noise=0.2)  
- Discuss:
  - Effect of L1 on sparsity and jagged decision boundaries  
  - Effect of L2 on smoothness and margin  
- Add class imbalance (70:30) in training set:
  - Report accuracy & AUROC  
  - Discuss impact of imbalance  

---

---

## **Question 3 — MNIST & CNN Experiments (3 marks)**

### **3.1 MLP on MNIST (1.5 marks)**
- Model:
  - 30-neuron layer → 20-neuron layer → 10-class output  
- Compare against:
  - Random Forest  
  - Logistic Regression  
- Report:
  - Accuracy  
  - F1-score  
  - Confusion matrix  
  - Observations + misclassification analysis  
- Visualize embeddings:
  - t-SNE of 20-neuron layer (trained vs untrained)  
- Cross-domain test:
  - Evaluate on **Fashion-MNIST**  
  - Compare t-SNE embeddings (MNIST vs Fashion-MNIST)  

---

### **3.2 CNN on MNIST (1.5 marks)**
- Implement a CNN:
  - Conv layer: 32 filters, 3×3  
  - MaxPool  
  - FC layer: 128 neurons  
  - Output layer: 10 neurons  
  - Activation: ReLU  
- Use **two pretrained CNNs** (e.g., AlexNet, MobileNet, EfficientNet) for inference.
- Compare all three:
  - Accuracy  
  - F1-score  
  - Confusion matrix  
  - Number of parameters  
  - Inference time on test set  

---

## **Submission Format**
- Provide **GitHub repository** with notebooks:
  - `question1.ipynb`
  - `question2.ipynb`
  - `question3.ipynb`
- Add written explanations inside notebooks.
- Include the **Streamlit app link** in Question 1 notebook.
