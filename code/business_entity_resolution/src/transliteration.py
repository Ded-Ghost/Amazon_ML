"""Word dictionary for Indian-script business names, learned from training pairs. Run:

    python -m src.transliteration        # -> cache/transliteration.tsv

About 10% of the India records in Source 2/3 write the name in an Indian
script ("राम मार्केटिंग प्राइवेट लिमिटेड"). unidecode turns that into rough
ASCII ("raam maarkettiNg praaivett limittedd") that shares few exact words
with the Latin spelling in Source 1 ("Ram Marketing Private Limited").

In the training ground truth, many such records are matched to a Latin S1
name with the same number of words, and the words are in the same order. So
we align the words by position and count which Latin word each Indian-script
word is paired with ("प्राइवेट" -> "private", "इंफोटेक" -> "infotech").
A word enters the dictionary when it was seen at least MIN_COUNT times and
at least MIN_SHARE of those times with the same Latin word.

The dictionary is learned only from the 80% training half of the S1
entities (never from validation) and only from the provided training data.
normalize.name_tokens() applies it to every name that contains non-Latin
characters, so blocking and all features see "private" instead of "praaivett".
"""
import sys
from collections import Counter

from .config import TRANSLITERATION_PATH as DICTIONARY_PATH, CACHE_DIR
from .data import load_ground_truth, load_source
from .normalize import NON_LATIN, clean, raw_words
from .split import split_s1_ids

MIN_COUNT = 3
MIN_SHARE = 0.5


def learn_dictionary(s1_names, match_names, gt, s1_ids):
    """{Indian-script word: Latin word} from position-aligned matched names."""
    pair_counts, word_counts = Counter(), Counter()
    for s1_id in s1_ids:
        latin = clean(s1_names[s1_id]).split()
        for match in gt[s1_id]:
            name = match_names[match]
            if not NON_LATIN.search(name):
                continue
            words = raw_words(name)
            if len(words) != len(latin):
                continue
            for word, latin_word in zip(words, latin):
                if NON_LATIN.search(word):
                    pair_counts[word, latin_word] += 1
                    word_counts[word] += 1
    best = {}
    for (word, latin_word), count in pair_counts.items():
        if count > best.get(word, ("", 0))[1]:
            best[word] = (latin_word, count)
    return {word: latin_word for word, (latin_word, count) in best.items()
            if word_counts[word] >= MIN_COUNT and count / word_counts[word] >= MIN_SHARE}


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # the examples contain Indian scripts
    gt = load_ground_truth()
    s1 = load_source("train", 1, usecols=["entity_id", "business_name"])
    train_ids, _ = split_s1_ids(s1["entity_id"])
    s1_names = dict(zip(s1["entity_id"], s1["business_name"]))
    match_names = {}
    for n in (2, 3):
        df = load_source("train", n, usecols=["entity_id", "business_name"])
        match_names.update(zip(df["entity_id"], df["business_name"]))
    dictionary = learn_dictionary(s1_names, match_names, gt, train_ids)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(DICTIONARY_PATH, "w", encoding="utf-8", newline="\n") as f:
        for word, latin_word in sorted(dictionary.items()):
            f.write(f"{word}\t{latin_word}\n")
    print(f"{len(dictionary):,} words -> {DICTIONARY_PATH}")
    print("examples:", list(dictionary.items())[:15])


if __name__ == "__main__":
    main()
