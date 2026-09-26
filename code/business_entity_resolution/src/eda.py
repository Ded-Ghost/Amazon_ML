"""Exploratory data analysis. Run from code/business_entity_resolution:

    python -m src.eda

Prints: row counts per source/country (train and test), data sanity checks,
singleton share, matches-per-entity distribution, S2 vs S3 share, country
agreement of matched pairs, empty-field rates, 20 example matched pairs and
10 French test records.
"""
import gc
import sys

import numpy as np
import pandas as pd

from .config import SEED
from .data import load_ground_truth, load_source

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def country_counts(split):
    """Rows per country (index) and source (columns) for one split."""
    counts = {}
    for n in (1, 2, 3):
        df = load_source(split, n, usecols=["entity_id", "country"])
        # sanity: ids unique and carrying the right prefix
        assert df["entity_id"].is_unique, f"{split} source{n}: duplicate ids"
        assert df["entity_id"].str.startswith(f"S{n}-").all(), f"{split} source{n}: bad prefix"
        counts[f"source{n}"] = df["country"].value_counts()
    table = pd.DataFrame(counts).fillna(0).astype(int)
    table["total"] = table.sum(axis=1)
    table.loc["ALL"] = table.sum()
    return table


def empty_field_rates(frames):
    """% of records with an empty name / address, per source and country."""
    rows = []
    for name, df in frames.items():
        for country, g in df.groupby("country"):
            rows.append({
                "source": name, "country": country, "rows": len(g),
                "% empty name": 100 * (g["business_name"].str.strip() == "").mean(),
                "% empty address": 100 * (g["business_address"].str.strip() == "").mean(),
            })
    return pd.DataFrame(rows).round(2)


def print_record(label, row):
    print(f"  {label} {row['entity_id']:<14} name: {row['business_name']}")
    print(f"  {'':{len(label)}} {'':<14} addr: {row['business_address']}  [{row['country']}]")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # names contain Devanagari etc.

    # ------------------------------------------------------------------ counts
    section("1. Row counts per source and country")
    for split in ("train", "test"):
        print(f"\n--- {split} ---")
        print(country_counts(split))

    # ------------------------------------------------------------ train load
    s1 = load_source("train", 1)
    s23 = pd.concat([load_source("train", 2), load_source("train", 3)], ignore_index=True)
    gt = load_ground_truth()

    section("2. Ground truth sanity checks")
    print(f"S1 entities: {len(s1):,}   ground-truth rows: {len(gt):,}")
    print(f"every S1 entity has a GT row: {set(gt) == set(s1['entity_id'])}")
    n_dupes = sum(len(v) != len(set(v)) for v in gt.values())
    print(f"GT rows with a repeated id: {n_dupes}")

    # one row per (S1, matched record) pair
    s1_ids = np.array(list(gt.keys()))
    n_matches = np.array([len(v) for v in gt.values()])
    pairs = pd.DataFrame({
        "s1_id": np.repeat(s1_ids, n_matches),
        "match_id": [m for v in gt.values() for m in v],
    })
    pairs["match_source"] = pairs["match_id"].str[:2]
    known = pairs["match_id"].isin(s23["entity_id"])
    print(f"matched pairs: {len(pairs):,}   pairs whose id is missing from S2/S3: {(~known).sum()}")
    print(f"prefixes in GT lists: {pairs['match_source'].value_counts().to_dict()}")

    # ------------------------------------------------------------ singletons
    section("3. Singletons and matches per S1 entity")
    per_entity = pd.DataFrame({"s1_id": s1_ids, "n_matches": n_matches})
    per_entity = per_entity.merge(s1[["entity_id", "country"]], left_on="s1_id", right_on="entity_id")
    singleton_frac = (per_entity["n_matches"] == 0).mean()
    print(f"singletons: {(per_entity['n_matches'] == 0).sum():,} of {len(per_entity):,} = {100 * singleton_frac:.2f}%")
    print("\nsingleton % by country:")
    print((per_entity.groupby("country")["n_matches"].apply(lambda s: 100 * (s == 0).mean())).round(2).to_string())

    capped = per_entity["n_matches"].clip(upper=10).astype(str).replace("10", "10+")
    dist = capped.value_counts()
    dist = dist.reindex([str(i) for i in range(10)] + ["10+"]).dropna().astype(int)
    print("\nmatches per S1 entity (count, % of entities):")
    print(pd.DataFrame({"entities": dist, "%": (100 * dist / len(per_entity)).round(2)}).to_string())
    m = per_entity.loc[per_entity["n_matches"] > 0, "n_matches"]
    print(f"\namong non-singletons: mean {m.mean():.2f}, median {m.median():.0f}, max {m.max()}")

    # ------------------------------------------------------------ S2 vs S3
    section("4. Where do matches come from? (S2 vs S3)")
    src_counts = pairs["match_source"].value_counts()
    print((100 * src_counts / src_counts.sum()).round(2).rename("% of matched pairs").to_string())

    per_src = pairs.groupby(["s1_id", "match_source"]).size().unstack(fill_value=0)
    print("\nnon-singleton S1 entities by which sources they match:")
    kinds = np.select(
        [(per_src["S2"] > 0) & (per_src["S3"] > 0), per_src["S2"] > 0],
        ["S2 and S3", "S2 only"], "S3 only",
    )
    print(pd.Series(kinds).value_counts().to_string())
    for src in ("S2", "S3"):
        c = per_src[src].clip(upper=5).astype(str).replace("5", "5+").value_counts().sort_index()
        print(f"\n# {src} matches per non-singleton entity:\n{c.to_string()}")

    per_record = pairs["match_id"].value_counts()
    print(f"\nS2/S3 records matched to more than one S1 entity: {(per_record > 1).sum():,}")
    for src in ("S2", "S3"):
        pool = s23["entity_id"].str.startswith(src).sum()
        used = (pairs["match_source"] == src).sum()
        print(f"{src}: {pool:,} records, {used:,} appear in GT ({100 * used / pool:.1f}%), "
              f"the rest are never matched (distractors)")

    # ------------------------------------------------------------ country
    section("5. Do matched records share the S1 country?")
    cpairs = (pairs
              .merge(s1[["entity_id", "country"]].rename(columns={"entity_id": "s1_id", "country": "s1_country"}), on="s1_id")
              .merge(s23[["entity_id", "country"]].rename(columns={"entity_id": "match_id", "country": "match_country"}), on="match_id"))
    same = (cpairs["s1_country"] == cpairs["match_country"]).mean()
    print(f"same country in {100 * same:.4f}% of {len(cpairs):,} matched pairs")
    print(pd.crosstab(cpairs["s1_country"], cpairs["match_country"]))

    # ------------------------------------------------------------ empty fields
    section("6. Empty name / address rates (train)")
    print(empty_field_rates({"S1": s1, "S2": s23[s23["entity_id"].str.startswith("S2")],
                             "S3": s23[s23["entity_id"].str.startswith("S3")]}).to_string(index=False))

    # ------------------------------------------------------------ examples
    section("7. 20 matched pairs (random, split evenly over the countries present)")
    countries = sorted(cpairs["s1_country"].unique())
    per_country = 20 // len(countries)
    s1_idx = s1.set_index("entity_id")
    s23_idx = s23.set_index("entity_id")
    for country in countries:
        sample = cpairs[cpairs["s1_country"] == country].sample(per_country, random_state=SEED)
        print(f"\n--- {country} ---")
        for _, p in sample.iterrows():
            print_record("S1 ", s1_idx.loc[p["s1_id"]].to_dict() | {"entity_id": p["s1_id"]})
            print_record(" ->", s23_idx.loc[p["match_id"]].to_dict() | {"entity_id": p["match_id"]})
            print()

    del s1, s23, s1_idx, s23_idx, pairs, cpairs, per_src, gt
    gc.collect()

    # ------------------------------------------------------------ test / France
    section("8. Test set: empty-field rates and 10 French records")
    test = {f"S{n}": load_source("test", n) for n in (1, 2, 3)}
    print(empty_field_rates(test).to_string(index=False))
    french = pd.concat([df[df["country"] == "France"] for df in test.values()], ignore_index=True)
    print(f"\nFrench test records: {len(french):,}\n")
    for _, row in french.sample(10, random_state=SEED).iterrows():
        print_record("   ", row)
        print()


if __name__ == "__main__":
    main()
