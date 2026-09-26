"""Central place for paths and the random seed.

Paths default to the layout of this repository:

    Amazon_ML/
    ├── student_resource/dataset/{train,test}/   <- data
    ├── output/                                  <- submission files
    └── code/business_entity_resolution/src/     <- this code

Both can be overridden with environment variables, so the code also runs
from the final submission zip (where the dataset lives somewhere else):

    BER_DATA_DIR=/path/to/dataset  BER_OUTPUT_DIR=/path/to/output
"""
import os
from pathlib import Path

SEED = 42

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # code/business_entity_resolution
REPO_ROOT = PACKAGE_ROOT.parents[1]                  # folder that holds code/ and output/

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", REPO_ROOT / "student_resource" / "dataset"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", REPO_ROOT / "output"))
CACHE_DIR = Path(os.environ.get("BER_CACHE_DIR", PACKAGE_ROOT / "cache"))  # intermediate results
TRANSLITERATION_PATH = CACHE_DIR / "transliteration.tsv"  # learned by src/transliteration.py

VAL_FRACTION = 0.2
