"""Architecture declarations and optional paper-recipe metadata.

The controlled study does not consume the paper recipes, but adapters still
query this module before selecting the controlled epoch and batch settings.
Keeping the registry here also prevents a model from being reported under the
wrong architecture name.
"""
from __future__ import annotations


EXPECTED = {
    "swininsseg": {
        "backbone": "Swin-T",
        "variant": "tiny",
        "detector": "SOLOv2",
        "mask_head": "SOLOv2 dynamic mask head",
        "attention": "MKA",
    },
    "solov2_gala": {
        "backbone": "ResNet-50",
        "detector": "SOLOv2",
        "mask_head": "SOLOv2 dynamic mask head",
        "attention": "GALA EFPN",
    },
}


PAPER_TRAINING = {
    # These entries document the reference defaults.  The controlled study
    # sets use_paper_training_config=False and therefore uses StudyConfig.
    "swininsseg": {},
    "solov2_gala": {},
}


def paper_training_config(model: str, dataset: str) -> dict:
    """Return a copy of optional model-paper training overrides."""
    if dataset not in {"ssdd", "hrsid", "visible"}:
        raise ValueError(f"Unknown dataset: {dataset}")
    return dict(PAPER_TRAINING.get(model, {}))


def check(model: str, actual: dict, strict: bool = True) -> dict:
    """Validate architecture fields that define the reported model name."""
    if not isinstance(actual, dict) or not actual:
        raise ValueError(f"No architecture declaration supplied for {model}")
    expected = EXPECTED.get(model)
    mismatches = {}
    if expected:
        mismatches = {
            key: {"expected": value, "actual": actual.get(key)}
            for key, value in expected.items()
            if actual.get(key) != value
        }
    if strict and mismatches:
        raise ValueError(f"Architecture mismatch for {model}: {mismatches}")
    return {**actual, "architecture_verified": not mismatches,
            "architecture_mismatches": mismatches}


def deviation_suffix(record: dict) -> str:
    """Make any permitted architecture deviation visible in result labels."""
    return "_deviation" if record.get("architecture_mismatches") else ""
