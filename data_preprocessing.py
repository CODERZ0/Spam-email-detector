"""
Shared text preprocessing for the Email/SMS Spam Detection system.

This module is used by both training and prediction to ensure
consistent text transformation.
"""

import string
import logging

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NLTK bootstrap – download required data once
# ---------------------------------------------------------------------------

def ensure_nltk_data() -> None:
    """Download NLTK resources if they are not already present."""
    # Workaround for SSL certificate issues on some systems (e.g. macOS)
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for resource in ("punkt", "punkt_tab", "stopwords"):
        try:
            nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
        except LookupError:
            logger.info("Downloading NLTK resource: %s", resource)
            nltk.download(resource, quiet=True)

ensure_nltk_data()

# Instantiate stemmer once at module level
_stemmer = PorterStemmer()
_stop_words = set(stopwords.words("english"))
_punctuation = set(string.punctuation)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def transform_text(text: str) -> str:
    """Clean and normalise a single text string.

    Steps
    -----
    1. Lower-case
    2. Tokenise (NLTK word_tokenize)
    3. Keep only alphanumeric tokens
    4. Remove English stop-words and punctuation characters
    5. Apply Porter stemming

    Parameters
    ----------
    text : str
        Raw email / SMS body text.

    Returns
    -------
    str
        Space-joined cleaned tokens.
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]

    # Remove stop-words and punctuation
    tokens = [t for t in tokens if t not in _stop_words and t not in _punctuation]

    # Stem
    tokens = [_stemmer.stem(t) for t in tokens]

    return " ".join(tokens)
