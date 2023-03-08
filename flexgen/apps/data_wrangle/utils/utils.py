# The source code in this file is mainly adapted from
# https://github.com/HazyResearch/fm_data_tasks/blob/main/fm_data_tasks/utils/utils.py
# which is under Apache License Version 2.0.

"""Misc utils."""
import logging
from pathlib import Path
from typing import List

from rich.logging import RichHandler


def setup_logger(log_dir: str):
    """Create log directory and logger."""
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    log_path = str(Path(log_dir) / "log.txt")
    handlers = [logging.FileHandler(log_path), RichHandler(rich_tracebacks=True)]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(module)s] [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def compute_metrics(preds: List, golds: List, task: str):
    """Compute metrics."""
    mets = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "crc": 0, "total": 0}
    for pred, label in zip(preds, golds):
        label = label.strip().lower()
        pred = pred.strip().lower()
        mets["total"] += 1
        if task in {
            "data_imputation",
            "entity_matching",
        }:
            crc = pred == label
        elif task in {"entity_matching", "schema_matching", "error_detection_spelling"}:
            crc = pred.startswith(label)
        elif task in {"error_detection"}:
            pred = pred.split("\n\n")[-1]
            breakpoint()
            crc = pred.endswith(label)
        else:
            raise ValueError(f"Unknown task: {task}")
        # Measure equal accuracy for generation
        if crc:
            mets["crc"] += 1
        if label == "yes":
            if crc:
                mets["tp"] += 1
            else:
                mets["fn"] += 1
        elif label == "no":
            if crc:
                mets["tn"] += 1
            else:
                mets["fp"] += 1

    prec = mets["tp"] / max(1, (mets["tp"] + mets["fp"]))
    rec = mets["tp"] / max(1, (mets["tp"] + mets["fn"]))
    acc = mets["crc"] / mets["total"]
    f1 = 2 * prec * rec / max(1, (prec + rec))
    return prec, rec, acc, f1