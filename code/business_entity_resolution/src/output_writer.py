"""Writing matching_results.tsv and candidate_pairs.tsv in the required format.

Format (both files): a header line, then exactly one row per Source 1 entity

    S1-00001<TAB>S2-00047,S2-00193,S3-00812
    S1-00003<TAB>                               <- empty list = no match

Rules enforced here (the writer raises instead of silently fixing, because a
violation means a bug upstream):
* every S1 id gets exactly one row, in the order given, and no unknown S1 ids
* only S2-/S3- ids in the lists (optionally: only ids that exist)
* no duplicate ids inside a list (duplicates are dropped, order kept)
* every matched id must also be a candidate
Files are UTF-8 with "\n" line endings, also on Windows.
"""
from pathlib import Path

MATCH_COLUMN = "matched_entity_ids"
CANDIDATE_COLUMN = "candidate_entity_ids"


def _clean_ids(s1_id, ids, valid_ids):
    ids = list(dict.fromkeys(ids))  # drop duplicates, keep order
    for entity_id in ids:
        if not entity_id.startswith(("S2-", "S3-")):
            raise ValueError(f"{s1_id}: {entity_id!r} is not an S2-/S3- id")
        if valid_ids is not None and entity_id not in valid_ids:
            raise ValueError(f"{s1_id}: {entity_id!r} does not exist in S2/S3")
    return ids


def write_id_list_tsv(path, s1_ids, id_lists, value_column, valid_ids=None):
    """Write one results-style TSV. id_lists: {s1_id: iterable of ids}."""
    s1_ids = list(s1_ids)
    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError("duplicate Source 1 ids")
    unknown = set(id_lists) - set(s1_ids)
    if unknown:
        raise ValueError(f"{len(unknown)} ids in the lists are not in Source 1, e.g. {sorted(unknown)[:3]}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{value_column}\n")
        for s1 in s1_ids:
            ids = _clean_ids(s1, id_lists.get(s1, ()), valid_ids)
            f.write(f"{s1}\t{','.join(ids)}\n")


def write_submission(output_dir, s1_ids, matches, candidates, valid_ids=None):
    """Write both files after checking that matches are a subset of candidates."""
    bad = [s1 for s1, ids in matches.items() if set(ids) - set(candidates.get(s1, ()))]
    if bad:
        raise ValueError(f"{len(bad)} entities have matches that are not candidates, e.g. {bad[:3]}")

    output_dir = Path(output_dir)
    write_id_list_tsv(output_dir / "matching_results.tsv", s1_ids, matches, MATCH_COLUMN, valid_ids)
    write_id_list_tsv(output_dir / "candidate_pairs.tsv", s1_ids, candidates, CANDIDATE_COLUMN, valid_ids)
