"""Lepton path helpers for the CLIP DEFT pipeline."""

_LEPTON_ROOT = "/workspace/users/sfarhatsabet"


def lepton_iter_dir(experiment_uuid: str, run_key: str) -> str:
    """Return the local Lepton directory holding one run's spec and results."""
    return f"{_LEPTON_ROOT}/{experiment_uuid}/{run_key}"
