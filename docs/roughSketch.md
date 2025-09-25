Below you’ll find:

1. A short high-level sequence you can follow.
2. Concrete, ordered tasks with checkpoints (what to do in each step).
3. Folder/repo structure and a minimal PyTorch starter sketch for the BiLSTM→LSTM seq2seq.
4. How to implement BPE / WordPiece from scratch (exact algorithm + simple reference code).
5. Experiment plan (which 3+ experiments to run and how to document them).
6. Streamlit deployment checklist and minimal app scaffold.
7. Evaluation & submission checklist (metrics, qualitative examples, blog).

---

# 1) High-level plan (one-line steps)

1. Download & inspect dataset (urdu_ghazals_rekhta).
2. Preprocess & build Urdu → Roman Urdu parallel pairs.
3. Implement tokenizers (your own BPE/WordPiece) and build vocabularies.
4. Build baseline rule-based transliterator (quick sanity check).
5. Implement PyTorch seq2seq: BiLSTM encoder (2 layers) + LSTM decoder (4 layers). Train + validate.
6. Run required experiments (3+), log results.
7. Evaluate (BLEU, perplexity, CER), collect qualitative examples.
8. Wrap in Streamlit UI and deploy public (Spaces/Heroku/Vercel/Streamlit Cloud).

---

# 2) Step-by-step tasks & checkpoints

## Step A — Setup (time: 0.5 day)

- Create a GitHub repo and a Colab notebook (or local venv).
- Required libs: `torch`, `torchtext` (optional), `sentencepiece` only for reference (but **don’t** use library tokenizers for assignment — implement from scratch), `numpy`, `tqdm`, `streamlit`. Document GPU resource if used.
- Commit a README that quotes the project rules (keeps academic integrity).

**Checkpoint A**: repo exists with README and skeleton directories (see below).

## Step B — Data acquisition & exploration (1 day)

- Clone dataset: [https://github.com/amir9ume/urdu_ghazals_rekhta](https://github.com/amir9ume/urdu_ghazals_rekhta) (the assignment points this).
- Inspect files: find Urdu script and transliteration columns. If Roman Urdu not directly present, you’ll need to derive transliteration rules (see below).
- Build a CSV of parallel pairs: `source (urdu_unicode)`, `target (roman_urdu)`.

**Checkpoint B**: `data/raw/` has original files and `data/pairs.csv` contains `urdu, roman` columns.

## Step C — Preprocessing & rule-based transliteration (2–3 days)

- Normalize Urdu:

  - Map variant characters to canonical ones (e.g., different forms of alef, normalizing hamza/ye/kashida, remove diacritics/tashkeel).
  - Clean punctuation and extra whitespace.

- If romanized target missing: create deterministic transliteration rules for common mappings (so you can bootstrap model training).

  - Example: `م`→`m`, `ا`→`a` or long/short rules, handle `ے, ی` etc.

- Create a small rule-based converter to create synthetic romanization for lines lacking it (document in notebook).

**Checkpoint C**: `data/processed/train.csv` + small sample of 100 manually-verified pairs.

(Requirement note: assignment expects you to derive or collect rules if roman not present.)

## Step D — Tokenization: Implement subword tokenizer (BPE or WordPiece) from scratch (2–3 days)

- Implement BPE or WordPiece **yourself** — libraries not allowed for tokenization. The doc hints BPE (sentencepiece or wordpiece) but requires custom implementation.
- I give full algorithm and short working code in Section 4 below.

**Checkpoint D**: functions `learn_bpe(corpus, merges)` and `encode_with_bpe(text)` are implemented and tested on sample lines.

## Step E — Data splits & data loaders (0.5 day)

- Splits: train 50%, val 25%, test 25% (per assignment). Shuffle before splitting.
- Save tokenized sequences and build PyTorch `Dataset`/`DataLoader`.

**Checkpoint E**: `data/splits/` contains `train.jsonl`, `val.jsonl`, `test.jsonl` (token ids + raw text).

## Step F — Build baseline model & sanity tests (1–2 days)

- Baseline 1: Rule-based transliterator output (evaluation vs. gold).
- Baseline 2: Simple character-level seq2seq (small model) to ensure training pipeline functional.

**Checkpoint F**: Baselines evaluated and results logged.

## Step G — Main model: BiLSTM encoder (2 layers) + LSTM decoder (4 layers) in PyTorch (3–5 days)

- Model details per assignment: encoder BiLSTM, decoder LSTM; implement attention (optional but recommended for performance) — not strictly required but helps.
- Loss: CrossEntropyLoss (ignore index for padding). Optimizer: Adam.
- Training loop: teacher forcing for decoder during training; validation after each epoch; checkpoint best model by BLEU/CER.

**Checkpoint G**: Trained model artifact and logs (training loss, val BLEU/perplexity).

## Step H — Experiments (2–4 days)

- At least **three** experiments varying parameters from the assignment (embedding dim, hidden size, encoder layers, decoder layers, dropout, lr, batch size). Suggested experiments below.
- Log each run with config, hyperparams, evaluation scores, and examples.

**Checkpoint H**: Experiments table (CSV/MD) with runs & metrics.

## Step I — Evaluation & qualitative analysis (1 day)

- Compute BLEU, perplexity, CER/Levenshtein. Show 10–20 examples: model output vs. ground truth.
- Save confusion/error analysis: common substitution errors.

**Checkpoint I**: `reports/evaluation.md` with metric tables and examples.

## Step J — Streamlit app + deploy (1–2 days)

- Build Streamlit UI: input Urdu, show predicted Roman Urdu, show ground truth if present, confidence metrics, examples.
- Deploy on Streamlit Cloud or Hugging Face Spaces; make public (assignment requirement).

**Checkpoint J**: public URL + deployment instructions in repo.

## Step K — Write blog & LinkedIn report (1 day)

- Write short blog (500–1000 words) describing dataset, preprocessing, tokenizer design, architecture, experiments, and results; link to repo and Streamlit demo. Tag instructor per instruction.

---

# 3) Repo / folder structure (recommended)

```
urdu2roman/
├─ data/
│  ├─ raw/                 # original dataset files
│  ├─ processed/           # cleaned text, tokenized files
│  └─ splits/              # train/val/test jsonl
├─ notebooks/
│  ├─ 01_explore.ipynb
│  ├─ 02_preprocess_and_rules.ipynb
│  └─ 03_tokenizer_bpe.ipynb
├─ src/
│  ├─ data.py              # Dataset, DataLoader builders
│  ├─ preprocess.py        # normalization and rule-based transliterator
│  ├─ bpe.py               # BPE / WordPiece implementation (from scratch)
│  ├─ model.py             # PyTorch BiLSTM encoder + LSTM decoder
│  ├─ train.py
│  └─ evaluate.py
├─ streamlit_app/
│  └─ app.py
├─ experiments/
│  └─ runs.csv
├─ reports/
│  └─ evaluation.md
├─ requirements.txt
└─ README.md
```

---

# 4) BPE (Byte Pair Encoding) / WordPiece implementation — from scratch

You must **not** use a library tokeniser. Below is the algorithm and a minimal Python implementation sketch you can drop into `src/bpe.py`. (This is small and practical — run it on target-side corpus to build vocab.)

### BPE algorithm (high level)

1. Build a vocabulary of words and their character sequences with an end-of-word marker (e.g., `l o w </w>`).
2. Count all symbol pair frequencies across the corpus.
3. Merge the most frequent pair into a single symbol (e.g., `l o` → `lo`).
4. Repeat merges `N` times or until vocab size target reached.
5. Encoding: greedily apply merges from largest units to produce subword tokens.

### Minimal BPE code (working skeleton)

```python
# src/bpe.py
from collections import Counter, defaultdict

def get_vocab(corpus_lines):
    vocab = Counter()
    for line in corpus_lines:
        for word in line.strip().split():
            chars = tuple(list(word) + ['</w>'])
            vocab[chars] += 1
    return vocab

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ''.join(pair)
    for word, freq in v_in.items():
        w = list(word)
        i = 0
        new_word = []
        while i < len(w):
            if i < len(w)-1 and w[i] == pair[0] and w[i+1] == pair[1]:
                new_word.append((w[i]+w[i+1]))
                i += 2
            else:
                new_word.append(w[i])
                i += 1
        v_out[tuple(new_word)] = freq
    return v_out

def learn_bpe(corpus_lines, num_merges=10000):
    vocab = get_vocab(corpus_lines)
    merges = []
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        vocab = merge_vocab(best, vocab)
    return merges

def encode_bpe(word, merges):
    # naive greedy: apply merges in order until no change
    w = list(word) + ['</w>']
    for a,b in merges:
        i=0
        new=[]
        while i < len(w):
            if i < len(w)-1 and w[i]==a and w[i+1]==b:
                new.append(a+b)
                i+=2
            else:
                new.append(w[i])
                i+=1
        w=new
    # remove end marker and return tokens
    if w and w[-1]=='</w>':
        w = w[:-1]
    return w
```

Notes:

- This is a simple implementation. Optimize/adjust for large corpora (use indexing, merges as string operations).
- WordPiece is similar but chooses merges differently (likelihood-based). If you prefer WordPiece, implement the likelihood-based merge scoring — but BPE is simpler and acceptable as the assignment hints BPE.

**Test** your BPE on the Roman Urdu target corpus (you can also train separate BPE for source Urdu script if you want — but usually use character-level or subword for source as well).

---

# 5) Minimal PyTorch seq2seq sketch (architecture constraints)

Below is a compact outline (to put in `src/model.py`) for a BiLSTM encoder and LSTM decoder. You can add attention later.

```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(input_size=emb_dim,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              dropout=dropout,
                              bidirectional=True,
                              batch_first=True)
    def forward(self, src, src_len):
        # src: (batch, seq)
        emb = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(emb, src_len, batch_first=True, enforce_sorted=False)
        outputs, (h, c) = self.bilstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # outputs: (batch, seq, hidden*2)
        # combine forward/back states
        # h: (num_layers*2, batch, hidden)
        return outputs, (h, c)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, num_layers=4, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden, teacher_forcing_ratio=0.5):
        # This is a simple looped decoder; for efficiency consider vectorized decoding during training
        batch_size = tgt.size(0)
        seq_len = tgt.size(1)
        outputs = []
        input_tok = tgt[:,0]  # assume <sos> at position 0
        h, c = hidden
        for t in range(1, seq_len):
            emb = self.embedding(input_tok).unsqueeze(1)  # (batch,1,emb)
            out, (h, c) = self.lstm(emb, (h, c))
            logits = self.out(out.squeeze(1))
            outputs.append(logits.unsqueeze(1))
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = logits.argmax(1)
            input_tok = tgt[:,t] if teacher_force else top1
        outputs = torch.cat(outputs, dim=1)
        return outputs
```

Notes:

- You must adapt shapes when combining BiLSTM encoder to decoder initial hidden states: combine forward/back hidden states (e.g., concatenate or sum and project to decoder `num_layers` and `hidden_size`) — implement a small projection layer to map `encoder_hidden` → `decoder_hidden`.
- Use `CrossEntropyLoss(ignore_index=pad_idx)` to compute loss.

---

# 6) Experiment matrix (what to run)

You need 3+ experiments. Suggested set (document each run with parameters):

1. **Exp A (baseline)**: emb=128, hidden=256, enc_layers=2, dec_layers=4, dropout=0.3, lr=1e-3, batch=64.
2. **Exp B (bigger embeddings)**: emb=256, hidden=512, enc_layers=2, dec_layers=4, dropout=0.3, lr=5e-4, batch=64.
3. **Exp C (deeper encoder)**: emb=256, hidden=256, enc_layers=4, dec_layers=4, dropout=0.3, lr=5e-4, batch=32.
4. (Optional) **Exp D (higher dropout / smaller lr)**: dropout=0.5, lr=1e-4, batch=128.

Log for each experiment:

- full config, training time/GPUs used, best val BLEU, test BLEU, perplexity, CER, 10 qualitative examples.

(Assignment requires at least three experiments.)

---

# 7) Evaluation metrics & how to compute

- BLEU: sentence-level or corpus-level via `nltk.translate.bleu_score` or your own implementation.
- Perplexity: exponentiate average cross-entropy per token.
- CER (Character Error Rate): use Levenshtein edit distance / normalized by ref length.
- Provide qualitative examples and error analysis (common mistakes, e.g., vowel missing, doubling, diacritics).

---

# 8) Streamlit app (minimal app scaffold)

File: `streamlit_app/app.py`

```py
import streamlit as st
from src.model import load_model, predict

st.title("Urdu → Roman Urdu Transliterator (BiLSTM NMT)")
input_text = st.text_area("Enter Urdu text:", height=150)
if st.button("Transliterate"):
    model = load_model("artifacts/best_model.pt")
    pred = predict(model, input_text)
    st.markdown("**Roman Urdu:**")
    st.write(pred)
```

- `load_model` should load tokenizer/vocab, model weights, and return a predict function which does normalization + tokenization + inference + detokenization.

Make sure this app is public (Streamlit Cloud or Hugging Face space) — assignment requires a public Streamlit cover.

---

# 9) Deliverables & README checklist (what to hand in)

- `src/` full code (preprocess, tokenizer, model, train, eval).
- `notebooks/` showing data exploration + tokenizer learning + sample outputs.
- `experiments/runs.csv` with hyperparams and metrics for each run.
- Trained model artifact(s) and instructions to reproduce results.
- Streamlit public link.
- Blog + LinkedIn post link(s).

---

# 10) Quick risks & tips

- **Data quality**: poetry has archaic/poetic forms; expect out-of-vocabulary. Use data augmentation (back-transliteration, noise) as bonus.
- **Tokenizer correctness**: test BPE merges visually and on held-out lines — bad merges can harm model.
- **Training time**: BiLSTM+LSTM with many layers can be slow — use smaller batches and monitor overfitting. Save checkpoints.
- **Evaluation**: BLEU can under-represent quality for romanizations (spelling variants). Use CER and human inspection.

---

# 11) Immediate next actions (concrete "what to do now" so you can start)

1. I will prepare (a) a Colab starter notebook that: downloads the repo, extracts Urdu & existing romanized lines, shows 50 random parallel pairs, and runs a tiny BPE (100 merges) so you can see tokens.
2. Also prepare `src/bpe.py` and `src/model.py` full working scripts (training loop + small example) ready to run in Colab/GPU.

Tell me: do you want the **Colab-ready notebook + runnable code** first (I’ll produce the code and you can run it), or would you like me to expand one particular section now (e.g., full train.py, full Streamlit app, or complete BPE implementation with tests)?

---
