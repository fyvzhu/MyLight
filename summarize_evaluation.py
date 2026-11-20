"""Summarize an existing evaluation run without loading a model or simulator."""
import argparse
from pathlib import Path

from evaluation_metrics import summarize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='Evaluation run containing config.yaml and data/*.csv')
    args = parser.parse_args()
    path = Path(args.run_dir).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    summarize_run(path)


if __name__ == '__main__':
    main()
