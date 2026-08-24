"""Per-token features for the SVM's NER tagger.

TF-IDF is a whole-document bag-of-words representation with no per-token
structure, so it can't feed a sequence tagger directly. This reframes NER as
per-token classification instead: a small hand-crafted feature vector per
word, the classical (pre-neural) answer to sequence tagging with a linear
classifier. sklearn.feature_extraction.DictVectorizer turns these dicts into
the sparse matrix LinearSVC needs.
"""

BOUNDARY = {"prev": "<s>", "next": "</s>"}


def token_features(words: list[str], i: int) -> dict:
    """One word's local context: the word itself, its shape, and its neighbors."""
    word = words[i]
    return {
        "word": word,
        "prefix3": word[:3],
        "suffix3": word[-3:],
        "word_len": len(word),
        "is_digit": word.isdigit(),
        "prev_word": words[i - 1] if i > 0 else BOUNDARY["prev"],
        "next_word": words[i + 1] if i < len(words) - 1 else BOUNDARY["next"],
        "position": i,
        "is_first": i == 0,
        "is_last": i == len(words) - 1,
    }
