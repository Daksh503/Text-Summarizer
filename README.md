# 🧠 Abstractive Text Summarization using Transformer Networks

## 📘 Description

This project implements an **Abstractive Text Summarization** system using a **Transformer-based encoder-decoder** architecture in PyTorch. The model learns to read lengthy input sequences and generate concise summaries that reflect the underlying meaning in rephrased language — emulating human-like summarization.

It demonstrates both foundational understanding and practical application of sequence-to-sequence modeling using self-attention mechanisms.

---

## 🧪 Technical Overview

### ✨ Model Architecture

The architecture follows the original **Transformer** design from Vaswani et al. (“Attention is All You Need”):

#### ➤ Encoder:
- **Token + Positional Embeddings**: Embeds input tokens with positional context.
- **Multi-head Self-Attention Layers**: Computes contextualized representations for each token.
- **Feed Forward Networks (FFN)**: Applies learned transformations to token-level features.
- **Layer Normalization + Residual Connections**: Improves gradient flow and stability.

#### ➤ Decoder:
- **Masked Multi-head Self-Attention**: Prevents peeking ahead during training.
- **Cross-Attention**: Attends to encoder outputs.
- **Feed Forward Network + Residuals**: Outputs transformed decoder states.
- **Linear + Softmax**: Projects to vocabulary distribution for token generation.

---

### 🔄 Forward Flow

1. Input article is tokenized and padded.
2. Tokens are embedded with positional encoding.
3. Encoder generates latent context vector sequences.
4. Decoder, given `<sos>` and previous outputs, generates tokens autoregressively using masked attention.
5. Greedy decoding is used for inference (can be extended to beam search).

---

## 🧼 Preprocessing Pipeline

- Lowercasing and punctuation removal
- Tokenization using `transformers`' tokenizer or custom tokenizer
- Padding and truncation to fixed length
- Special tokens: `<sos>`, `<eos>`, `<pad>`, `<unk>`

This step ensures uniform batch processing and compatibility with attention masking.

---

## 🧰 Core Dependencies

```bash
pip install torch transformers
