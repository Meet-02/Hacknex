"""
evaluation/dataset.py
Labeled evaluation dataset for retrieval tuning.

Each entry:
{
    "query"          : str,
    "expected_camera": str | None,
    "time_start"     : float | None,
    "time_end"       : float | None,
    "is_negative"    : bool,   # True = no match in footage
}
Add more entries as real footage is collected.
Include negatives so the no-match threshold is tuned on both sides.
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Dict, Any, Optional

SEED_DATASET: List[Dict[str, Any]] = [
    # --- Positive queries ---
    {
        "query"          : "red car entering the main gate",
        "expected_camera": "G336",
        "time_start"     : None,
        "time_end"       : None,
        "is_negative"    : False,
    },
    {
        "query"          : "person carrying a bag at the lobby",
        "expected_camera": "G338",
        "time_start"     : None,
        "time_end"       : None,
        "is_negative"    : False,
    },
    {
        "query"          : "truck near side entrance after 2 pm",
        "expected_camera": "G339",
        "time_start"     : 50400.0,
        "time_end"       : None,
        "is_negative"    : False,
    },
    {
        "query"          : "white car in parking lot",
        "expected_camera": "G337",
        "time_start"     : None,
        "time_end"       : None,
        "is_negative"    : False,
    },
    # --- Negative queries (nothing matching should be in footage) ---
    {
        "query"          : "yellow helicopter landing on the roof",
        "expected_camera": None,
        "time_start"     : None,
        "time_end"       : None,
        "is_negative"    : True,
    },
    {
        "query"          : "person wearing a purple jacket",
        "expected_camera": None,
        "time_start"     : None,
        "time_end"       : None,
        "is_negative"    : True,
    },
]


def load_dataset(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load from JSON file if path given, else return the seed set."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return SEED_DATASET


def save_dataset(dataset: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[dataset] Saved {len(dataset)} entries to {path}")
