# Statistical N-Gram Language Models

This repository contains two distinct implementations of statistical language models, demonstrating the progression from a basic trigram character/word frequency model to a more robust, adaptive interpolated n-gram model using subword tokenization.

---

## 1. Basic Trigram Statistical Model 
**File:** `trigram_statistical_model_implementation.py`

This script provides a foundational implementation of a trigram language model using a small, hardcoded corpus. It leverages PyTorch tensors to build a 3D frequency matrix, which is then normalized into a probability distribution. 

### Key Features:
* **Custom Vocabulary Building:** Manually tokenizes text and builds string-to-index (`stoi`) and index-to-string (`itos`) mappings.
* **Tensor-Based Counting:** Uses a 3D PyTorch tensor (`torch.zeros((vocab_size, vocab_size, vocab_size))`) to track the occurrences of word triplets $(w_1, w_2, w_3)$.
* **Text Generation:** Samples from the normalized probability distributions using `torch.multinomial` to generate new, synthetic sentences based on the learned trigram distributions.
* **Evaluation:** Calculates the average log-loss (negative log-likelihood) of the training text to evaluate model fit.

### Dependencies:
* `torch`
* `matplotlib`

---

## 2. Interpolated N-Gram Model with Adaptive Weights
**File:** `interpolated_n_gram_model.py`

This script builds a more advanced, production-style statistical model. Instead of relying solely on trigrams (which suffer heavily from data sparsity), it linearly interpolates unigram, bigram, and trigram probabilities. It trains adaptive weights for this interpolation using the Expectation-Maximization (EM) algorithm.

### Key Features:
* **Real-World Data:** Automatically downloads and utilizes a slice of the **Text8** dataset (Wikipedia text).
* **Subword Tokenization:** Integrates `sentencepiece` (Byte-Pair Encoding) instead of basic whitespace splitting, allowing it to handle out-of-vocabulary words gracefully.
* **Adaptive Interpolation:** Groups bigram contexts into frequency bins and uses the EM algorithm on a holdout validation set to learn the optimal interpolation weights ($\alpha_1, \alpha_2, \alpha_3$) for each bin. This ensures the model relies on trigrams when data is abundant, and falls back to bigrams/unigrams when data is sparse.
* **Next-Token Prediction:** Given a two-token context (e.g., "king sit"), it calculates the interpolated probabilities across the entire vocabulary and returns the top 5 most likely next tokens.

### Dependencies:
* `sentencepiece`
* `numpy`
* `tqdm`

---

## Installation & Setup

Ensure you have Python installed, then install the required dependencies for both scripts:

```bash
pip install torch matplotlib sentencepiece numpy tqdm
