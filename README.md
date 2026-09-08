# SVM Baseline — Tweet Classification (NLP Project)

This module implements the **TF-IDF + Linear SVM** baseline for the group's NLP assignment. The assignment requires each member to implement a different NLP method on a shared task; this repo's other two models (BiLSTM, RoBERTa-CNN) live alongside this one and are evaluated on the same data split for a fair comparison.

The chosen NLP problem is **tweet classification** across four sub-tasks:

| Task | Type | Approach |
|---|---|---|
| Sentiment analysis | Single-label multiclass | TF-IDF + `LinearSVC` |
| Emotion detection | Single-label multiclass | TF-IDF + `LinearSVC` |
| Topic classification | Single-label multiclass | TF-IDF + `LinearSVC` |
| Named entity recognition | Per-token BIO tagging | Hand-crafted token features + `LinearSVC` |

All four are independent linear SVMs — there is no shared training, no gradient sharing, and no multi-task loss. The TF-IDF vectorizer is shared only across the three document-level tasks (sentiment, emotion, topic), since they all operate on the same whole-tweet bag-of-words representation. NER cannot use that representation and is handled separately (see below).

## Why a linear SVM

Per the assignment's suggested methods for text classification (Naïve Bayes, SVM, or transformer-based models), this module implements the classical linear-model baseline: `TfidfVectorizer` for feature extraction and `LinearSVC` for classification, run once for each task, contrasted against the group's neural (BiLSTM) and transformer-based (RoBERTa-CNN) implementations.

## Pipeline

Text goes through two stages before it reaches this module, then three stages within it:

```
merged_tweets.csv
      │
      ▼
base_cleaning.py     — repair mojibake, decode HTML entities, strip "RT @user:",
                        drop empty/duplicate rows, keep lang == "en",
                        assign train/val/test split (splits.py)
      │
      ▼
cleaned_tweets.csv    — the row set every model in the project trains on
      │
      ▼
preprocess_svm.py     — SVM-specific text cleaning (see below)
      │
      ▼
svm_input.csv          — this module's model-ready input
      │
      ├─► data.py + model.py   — TF-IDF → LinearSVC, one per task (sentiment/emotion/topic)
      └─► ner_bio.py + ner_features.py + model.py — per-token features → LinearSVC (NER)
      │
      ▼
train.py → saved artifacts + evaluate.py → metrics.txt / metrics.json
      │
      ▼
predict.py — inference on a new tweet
```

Preprocessing for this module is not a copy of `base_cleaning.py` — it is a second, SVM-specific pass (`preprocess_svm.py`) built for a bag-of-words model:

1. `remove_links` — strip URLs, which add no signal for classification.
2. `remove_unicode_punctuation` — a Unicode-aware pass run *before* demojizing, since it removes curly quotes Twitter inserts (e.g. `it's` vs `it’s`) that would otherwise split the vocabulary into two features.
3. `demojize_to_token` — converts each emoji into a single underscore-joined token (e.g. `❤️` → `red_heart`) so it survives TF-IDF as a distinct feature.
4. `lowercase`
5. `remove_punctuation` — an ASCII-only pass run *after* demojizing, so it doesn't strip the underscore out of tokens like `soccer_ball`.
6. `normalize_whitespace`

The step order matters: unicode punctuation is stripped before emoji conversion so it doesn't interfere with emoji detection, and ASCII punctuation is stripped after emoji conversion so it doesn't break the emoji tokens it just created.

This pipeline only rewrites the `tweet` column — row count, `id`, and `split` are left untouched, which is what lets all three models in the project be compared on one shared test set.

## Data split

Splits are computed once in `splits.py` and reused across the whole project via `data/processed/splits.csv`, keyed by tweet `id`:

- **70% train / 15% validation / 15% test**, seeded (`SEED = 42`).
- Built with `GroupShuffleSplit`, grouped on a near-duplicate signature (`utils.group_key`) so that near-identical tweets never end up split across train and test.
- Not stratified — grouping and stratifying can't both be enforced by `GroupShuffleSplit` — but per-split class balance is printed by `splits.summarise()` and drifts by under 1.5 percentage points on this dataset.

This module's `data.load_and_split()` only uses the **train** and **test** partitions. The **validation** slice is loaded but unused here, since `LinearSVC` has no iterative training loop and therefore no early-stopping criterion to validate against; it exists in `splits.csv` for the BiLSTM and RoBERTa-CNN models, which do use it.

## Document-level tasks (sentiment, emotion, topic)

- **Features:** `TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), min_df=2, max_df=0.95, sublinear_tf=True)`, fit once on the training tweets and reused for all three tasks.
- **Model:** one `LinearSVC` per task, trained independently on the same TF-IDF matrix with task-specific labels.
- **Class imbalance:** inverse-frequency sample weights, computed per task, capped at **10×** (`WEIGHT_CAP` in `model.py`) so a rare class can't dominate the loss.
- **Labels:** encoded with a `LabelEncoder` per task, fit on the training split only.

## Named entity recognition

TF-IDF is a whole-document representation with no notion of word order or position, so it cannot feed a sequence tagger directly. NER is reframed as **per-token classification with a BIO tagging scheme**:

1. `ner_bio.add_bio_tags()` parses the dataset's `ner` column (entities as `TYPE: name, name | TYPE: name`) and locates each entity's words inside the tweet's own word list, tagging the first word `B-<TYPE>` and any following words `I-<TYPE>` (untagged words get `O`). Longer entities are matched before shorter ones so a multi-word entity isn't shadowed by a substring match.
2. `ner_features.token_features()` builds a small hand-crafted feature dictionary per word: the word itself, its first/last three characters, its length, whether it's numeric, its left/right neighbor, its position in the tweet, and whether it's the first/last word.
3. `sklearn.feature_extraction.DictVectorizer` converts those dictionaries into a sparse matrix, which trains one class-weighted `LinearSVC` (`model.train_ner_tagger`) over all BIO tag classes at once — a single multiclass tagger, not one classifier per entity type.

At inference (`predict.tag_words`), the same feature pipeline runs over the cleaned tweet's words and the tagger predicts one BIO tag per word.

At evaluation time (`evaluate._evaluate_ner`), the tagger is scored two ways: token-level BIO accuracy/F1 (this model's own diagnostic, dominated by the `O` class) and entity-type presence, which collapses predicted BIO tags to a per-tweet set of entity types found (`types_from_bio`) and compares that against the gold entities — the shared metric all three models in the project report on, for a fair NER comparison.

## GPU / CPU backend

`model.py` tries to import RAPIDS `cuml`/`cupy` at module load. If a CUDA GPU is detected, `TfidfVectorizer` and `LinearSVC` (for sentiment/emotion/topic) run on GPU; otherwise the code transparently falls back to the scikit-learn equivalents. The NER tagger always uses scikit-learn's `DictVectorizer` and `LinearSVC`, since cuML has no `DictVectorizer`. `model.to_numpy()` moves GPU arrays back to host memory wherever CPU-side code (evaluation, `LabelEncoder`, saving) needs a NumPy array.

## Repository layout

```
config.py            Paths, task list, TF-IDF hyperparameters, artifact filenames
data.py               Load/split data, build label encoders, build TF-IDF features, build NER token dataset
model.py              GPU/CPU backend selection, vectorizer/model construction, class weighting, training
train.py              End-to-end training entry point; saves artifacts, metrics, and predictions
evaluate.py           Scores a trained model on the test split (per-task + NER)
predict.py            Loads saved artifacts and runs inference on new tweet text
ner_bio.py            Converts the dataset's entity annotations into per-word BIO tags
ner_features.py       Per-token feature dictionary for the NER tagger
preprocess_svm.py     SVM-specific text cleaning pipeline
pipeline.py           Shared runner: applies a cleaning step chain and writes the model-ready CSV
base_cleaning.py      Project-wide cleaning + dedup + language filter + split assignment (shared by all models)
splits.py             Builds/persists the shared 70/15/15 train/val/test split
utils.py              Atomic, pure text-cleaning helper functions
```

## Saved artifacts

`train.py` writes four files to the model directory:

| File | Contents |
|---|---|
| `tfidf.pkl` | Fitted `TfidfVectorizer` |
| `label_encoders.pkl` | `LabelEncoder` per document-level task |
| `ner_vectorizer.pkl` | Fitted `DictVectorizer` for NER token features |
| `svm_models.pkl` | Dict of trained `LinearSVC` models (one per task, plus the NER tagger and its tag label list) |

`predict.py` looks for these first in the current model directory and falls back to a legacy location from earlier runs, so a machine that hasn't retrained can still serve predictions from an older artifact set.

## Setup

```bash
pip install scikit-learn pandas numpy joblib emoji ftfy
# optional, for GPU acceleration:
pip install cudf-cu12 cuml-cu12 cupy-cuda12x
```

## Usage

Run each stage in order the first time (or whenever the merged dataset changes):

```bash
# 1. Project-wide cleaning + split assignment (shared by all three models)
python -m src.data_cleaning.base_cleaning

# 2. SVM-specific text cleaning
python -m src.data_cleaning.preprocess_svm

# 3. Train all four SVM models and evaluate on the test split
python -m src.models.svm.train

# 4. Re-run evaluation only, from saved artifacts
python -m src.models.svm.evaluate

# 5. Predict on a new tweet
python -m src.models.svm.predict "Messi just won the World Cup with Argentina!"
```

`train.py` prints per-task class counts, TF-IDF vocabulary size, NER tag vocabulary, and training token count, then writes `metrics.txt`, `metrics.json`, and `predictions.csv` to the model directory alongside the artifacts above.

## Known limitations

- `cuml`-pickled artifacts may fail to load on a machine without `cuml` installed; a compatibility guard in `predict.py`'s artifact loading is worth adding if the group trains on GPU but serves predictions on CPU.
- NER entity matching in `ner_bio.tag_sentence` is exact-string matching against the cleaned tweet's words, so entities whose surface form doesn't survive cleaning intact (e.g. altered by emoji/punctuation stripping) won't be tagged, which is reflected in the "entities matched" rate `ner_bio.add_bio_tags` prints.
- Per-split class balance is not perfectly stratified (see Data split above); acceptable on this dataset but worth re-checking if the dataset changes materially.
