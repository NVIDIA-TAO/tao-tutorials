"""Command-line argument parsing for the DEFT pipeline entry points."""

import argparse


def parse_args():
    """Parse command-line arguments for the DEFT pipeline."""
    parser = argparse.ArgumentParser(
        description="PAS-DEFT Pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the pipeline configuration YAML file"
    )

    return parser.parse_args()
