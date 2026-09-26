"""Text cleaning shared by blocking and matching.

clean():           any script -> ASCII (unidecode), lowercase, "&" -> "and",
                   apostrophes removed, other punctuation -> space.
name_tokens():     clean() + undo digit-for-letter swaps ("y0ga" -> "yoga")
address_tokens():  clean() + split letters from numbers ("1604b" -> "1604 b")
                   + drop leading zeros ("012" -> "12")
skeleton():        a rough phonetic key for one token: drop vowels (and h/y),
                   merge letters that sound alike, collapse repeats. It makes
                   spelling variants, vowel typos and transliterations collide:
                   "private" / "praaivett" -> "prvt", "limited" / "limittedd" -> "lmt".

If cache/transliteration.tsv exists (learned from training pairs by
src/transliteration.py), name_tokens() first replaces known Indian-script
words by their Latin spelling ("प्राइवेट" -> "private"). Nothing else here is
specific to a country or language.
"""
import re

from unidecode import unidecode

from .config import TRANSLITERATION_PATH

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")
_ORDINAL = re.compile(r"^[0-9]+(st|nd|rd|th)$")
_SKELETON_DROP = re.compile(r"[aeiouyh]")
_REPEATS = re.compile(r"(.)\1+")
_SOUND_ALIKE = str.maketrans({"c": "k", "q": "k", "g": "k", "z": "s", "x": "s",
                              "d": "t", "b": "p", "w": "v"})
_DIGIT_AS_LETTER = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t"})
NON_LATIN = re.compile(r"[^\x00-\x7FÀ-ɏ]")
_WORD = re.compile(r"[\wऀ-෿]+")  # \w misses Indic vowel signs, so add the Indic blocks


def raw_words(text):
    """Words of the original (not transliterated) text, e.g. ["राम", "मार्केटिंग"]."""
    return _WORD.findall(text)


def _load_transliteration():
    if not TRANSLITERATION_PATH.exists():
        return {}
    with open(TRANSLITERATION_PATH, encoding="utf-8") as f:
        return dict(line.rstrip("\n").split("\t") for line in f if line.strip())


_TRANSLITERATION = _load_transliteration()


def clean(text):
    text = unidecode(text).lower().replace("&", " and ").replace("'", "")
    return _NON_ALNUM.sub(" ", text).strip()


def _strip_zeros(token):
    return (token.lstrip("0") or "0") if token.isdigit() else token


def name_tokens(name):
    if _TRANSLITERATION and NON_LATIN.search(name):
        name = _WORD.sub(lambda m: _TRANSLITERATION.get(m.group(), m.group()), name)
    tokens = []
    for t in clean(name).split():
        # a word with >= 2 letters and some digits is usually a typo like "y0ga"
        if not t.isalpha() and sum(c.isalpha() for c in t) >= 2 and not _ORDINAL.match(t):
            t = t.translate(_DIGIT_AS_LETTER)
        tokens.append(_strip_zeros(t))
    return tokens


def address_tokens(address):
    return [_strip_zeros(t) for t in _LETTER_DIGIT.sub(" ", clean(address)).split()]


def skeleton(token):
    """Phonetic key of an alphabetic token, or "" when too short to be useful."""
    if not token.isalpha():
        return ""
    key = token.replace("ph", "f").translate(_SOUND_ALIKE)
    key = _REPEATS.sub(r"\1", _SKELETON_DROP.sub("", key))
    return key if len(key) >= 2 else ""
