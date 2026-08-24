"""Atomic text-cleaning helpers.

Every function here is pure: takes a string, returns a string, no pandas.
Pipelines in preprocess_*.py compose these into model-specific chains.
"""

import html
import re
import string
import unicodedata

import emoji
import ftfy

_URL_RE = re.compile(r"https?://\S+|www\.\S+|\bhttp\b")
_TCO_URL_RE = re.compile(r"https?://t\.co/\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_SYMBOL_RE = re.compile(r"#(\w+)")
_NUMBER_RE = re.compile(r"\d+")
_WHITESPACE_RE = re.compile(r"\s+")
_RT_PREFIX_RE = re.compile(r"^RT\s+@\w+:\s*")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def fix_mojibake(text: str) -> str:
    return ftfy.fix_encoding(text)


def unescape_html(text: str) -> str:
    """Decode HTML entities Twitter leaves in raw text, e.g. '&amp;' -> '&',
    '&lt;3' -> '<3'. Without this, models see literal 'amp'/'lt' tokens."""
    return html.unescape(text)


def remove_links(text: str) -> str:
    return _URL_RE.sub("", text)

def remove_tco_links(text: str) -> str:
    """Strip Twitter t.co URLs while leaving other web links intact."""
    return _TCO_URL_RE.sub("", text)


def replace_links_with_token(text: str, token: str = "http") -> str:
    """Replace URLs with a placeholder instead of deleting them.
    links normalized to the literal string 'http' rather than removed
    """
    return _URL_RE.sub(token, text)


def remove_rt_prefix(text: str) -> str:
    """Strip the leading 'RT @user:' marker from retweets."""
    return _RT_PREFIX_RE.sub("", text)


def remove_mentions(text: str) -> str:
    return _MENTION_RE.sub("", text)


def replace_mentions_with_token(text: str, token: str = "@user") -> str:
    """Replace @mentions with a generic token instead of deleting them."""
    return _MENTION_RE.sub(token, text)


def remove_hashtag_symbol(text: str) -> str:
    """Keep the hashtag word, drop only the '#' (e.g. '#WorldCup' -> 'WorldCup')."""
    return _HASHTAG_SYMBOL_RE.sub(r"\1", text)


def demojize_to_token(text: str, language: str = "en") -> str:
    """Convert each emoji into a single vocabulary token, e.g. '❤️' -> ' red_heart '."""
    return emoji.demojize(text, delimiters=(" ", " "), language=language)


def lowercase(text: str) -> str:
    return text.lower()


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_unicode_punctuation(text: str) -> str:
    """Strip all Unicode punctuation characters from a string.

    Removes all characters belonging to Unicode punctuation categories (P*),
    including non-ASCII variants such as curly quotes, em dashes, and ellipses,
    while preserving symbols (e.g., emoji in category 'So') and alphanumeric tokens.
    """
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def remove_numbers(text: str) -> str:
    return _NUMBER_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/newlines/tabs into single spaces and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def canonical_key(text: str) -> str:
    """Reduce a tweet to a model-independent identity string.

    Decodes HTML entities, lowercases, drops URLs and @mentions, replaces every
    non-alphanumeric character with a space, and collapses whitespace. Two
    tweets differing only in case, punctuation, links or mentions produce the
    same key.

    Used by base_cleaning.py to decide which rows survive deduplication, and by
    metrics.py to match entity names against tweet text. Never used as model
    input — the surviving row keeps its original 'tweet' text.

        canonical_key("RT @fifa: MESSI scores!! https://t.co/x")  ->  "messi scores"
    """
    text = unescape_html(str(text)).lower()
    text = remove_links(text)
    text = remove_mentions(text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return normalize_whitespace(text)


def group_key(canonical: str) -> str:
    """Reduce a canonical key to a near-duplicate cluster signature.

    Takes the tweet's content words (longer than two characters), deduplicates
    them and sorts them, so tweets with the same words in any order — or
    differing only by short filler words — map to the same signature. Falls back
    to the canonical key when no word qualifies.

    splits.py keeps rows sharing a signature in the same train/val/test split.

        group_key("messi scores again")  ->  "again messi scores"
        group_key("again messi scores")  ->  "again messi scores"
    """
    words = sorted({word for word in canonical.split() if len(word) > 2})
    return " ".join(words) or canonical
