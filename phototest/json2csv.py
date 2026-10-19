#!/usr/bin/env python3
"""Convert JSON / JSONL files to CSV.

Usage:
    python json2csv.py input.json output.csv
    python json2csv.py input.jsonl output.csv
    python json2csv.py input.jsonl output.csv --delimiter ";"

- .json  input can be a single object or a list of objects.
- .jsonl input must be one JSON object per line.
- Nested dicts are flattened into dot-separated columns (e.g. metadata.eyes.eyeshadow_tone).
- Lists are JSON-encoded as a single cell value.
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def flatten(obj, prefix="", sep="."):
    """Flatten a nested dict into a single-level dict with dot-separated keys."""
    items = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{prefix}{sep}{key}" if prefix else str(key)
            items.update(flatten(value, new_key, sep))
    elif isinstance(obj, list):
        items[prefix] = json.dumps(obj, ensure_ascii=False)
    else:
        items[prefix] = obj
    return items


def load_records(input_path: Path):
    text = input_path.read_text(encoding="utf-8")
    if input_path.suffix.lower() == ".jsonl":
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
        return records

    data = json.loads(text)
    if isinstance(data, list):
        return data
    return [data]


def convert(input_path: Path, output_path: Path, delimiter: str = ","):
    records = load_records(input_path)
    if not records:
        raise ValueError("No records found in input file.")

    flat_records = [flatten(r) if isinstance(r, dict) else {"value": r} for r in records]

    fieldnames = []
    seen = set()
    for record in flat_records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore")
        writer.writeheader()
        for record in flat_records:
            writer.writerow(record)

    print(f"Wrote {len(flat_records)} rows, {len(fieldnames)} columns -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert JSON/JSONL to CSV.")
    parser.add_argument("input", type=Path, help="Path to input .json or .jsonl file")
    parser.add_argument("output", type=Path, help="Path to output .csv file")
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        convert(args.input, args.output, args.delimiter)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
