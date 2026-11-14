import csv
import os


class CSVLogger:
    """Append rows of metrics to a CSV file (one row per call)."""

    def __init__(self, path: str, fieldnames: list):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def log(self, row: dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)