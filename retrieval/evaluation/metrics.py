"""
evaluation/metrics.py
Retrieval evaluation metrics:
    Recall@1/5/10, MRR, camera accuracy, timestamp accuracy, latency.
Also: threshold calibration helper (grid-search NO_MATCH_THRESHOLD).
"""

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Dict, Any, Optional, Callable
import config
from evaluation.dataset import load_dataset


# ------------------------------------------------------------------
# Core metric functions
# ------------------------------------------------------------------

def recall_at_k(results: Dict, expected_camera: str, k: int) -> float:
    """1.0 if expected_camera is in top-k evidence."""
    if results.get("status") != "verified":
        return 0.0
    top_k = results["evidence"][:k]
    return float(any(e["camera_id"] == expected_camera for e in top_k))


def reciprocal_rank(results: Dict, expected_camera: str) -> float:
    """1/rank of first correct camera hit, 0 if not found."""
    if results.get("status") != "verified":
        return 0.0
    for i, e in enumerate(results["evidence"], start=1):
        if e["camera_id"] == expected_camera:
            return 1.0 / i
    return 0.0


def camera_accuracy(results: Dict, expected_camera: str) -> float:
    return recall_at_k(results, expected_camera, k=1)


def timestamp_accuracy(
    results: Dict,
    expected_ts: float,
    tolerance: float = config.TIMESTAMP_TOLERANCE_SEC,
) -> float:
    """Top-1 timestamp within tolerance of expected?"""
    if results.get("status") != "verified" or not results["evidence"]:
        return 0.0
    ts = results["evidence"][0].get("timestamp_sec")
    if ts is None:
        return 0.0
    return float(abs(ts - expected_ts) <= tolerance)


def no_match_accuracy(results: Dict, is_negative: bool) -> float:
    """Correct if system matched polarity (verified vs no_verified_match)."""
    if is_negative:
        return float(results.get("status") == "no_verified_match")
    return float(results.get("status") == "verified")


# ------------------------------------------------------------------
# Full evaluation run
# ------------------------------------------------------------------

def evaluate(
    search_fn: Callable,
    dataset: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run full evaluation.
    search_fn: callable(query: str) -> envelope dict.
    """
    if dataset is None:
        dataset = load_dataset()

    r1, r5, r10, mrr = [], [], [], []
    cam_acc, nm_acc, latencies = [], [], []

    for entry in dataset:
        query   = entry["query"]
        exp_cam = entry.get("expected_camera")
        is_neg  = entry.get("is_negative", False)

        t0      = time.perf_counter()
        results = search_fn(query)
        lat     = time.perf_counter() - t0
        latencies.append(lat)

        nm_acc.append(no_match_accuracy(results, is_neg))

        if not is_neg and exp_cam:
            r1.append(recall_at_k(results, exp_cam, 1))
            r5.append(recall_at_k(results, exp_cam, 5))
            r10.append(recall_at_k(results, exp_cam, 10))
            mrr.append(reciprocal_rank(results, exp_cam))
            cam_acc.append(camera_accuracy(results, exp_cam))

        if verbose:
            print(f"  [{results.get('status')}] {query[:55]:<55} {lat*1000:.0f}ms")

    def _mean(lst):
        return float(np.mean(lst)) if lst else float("nan")

    metrics = {
        "Recall@1"       : _mean(r1),
        "Recall@5"       : _mean(r5),
        "Recall@10"      : _mean(r10),
        "MRR"            : _mean(mrr),
        "CameraAccuracy" : _mean(cam_acc),
        "NoMatchAccuracy": _mean(nm_acc),
        "LatencyMean_ms" : _mean(latencies) * 1000,
        "LatencyP95_ms"  : float(np.percentile(latencies, 95)) * 1000
                           if latencies else float("nan"),
        "n_positive"     : len(r1),
        "n_negative"     : sum(1 for e in dataset if e.get("is_negative")),
    }

    if verbose:
        print("\n=== Retrieval Metrics ===")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<22}: {v:.4f}")
            else:
                print(f"  {k:<22}: {v}")

    return metrics


# ------------------------------------------------------------------
# Threshold calibration
# ------------------------------------------------------------------

def calibrate_threshold(
    search_fn: Callable,
    dataset: Optional[List[Dict]] = None,
    thresholds: Optional[List[float]] = None,
) -> float:
    """
    Grid-search NO_MATCH_THRESHOLD to maximise no-match accuracy.
    Updates config.NO_MATCH_THRESHOLD in memory — persist manually.
    """
    if dataset is None:
        dataset = load_dataset()
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.90, 0.05)]

    best_th, best_score = config.NO_MATCH_THRESHOLD, -1.0
    for th in thresholds:
        config.NO_MATCH_THRESHOLD = th
        scores = [
            no_match_accuracy(search_fn(e["query"]), e.get("is_negative", False))
            for e in dataset
        ]
        avg = float(np.mean(scores))
        print(f"  threshold={th:.2f}  no-match-acc={avg:.4f}")
        if avg > best_score:
            best_score, best_th = avg, th

    config.NO_MATCH_THRESHOLD = best_th
    print(f"\nBest threshold: {best_th:.2f}  (no-match-acc={best_score:.4f})")
    return best_th


if __name__ == "__main__":
    from retrieval.search import search
    evaluate(search)
