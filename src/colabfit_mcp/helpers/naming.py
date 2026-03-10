from datetime import datetime
from pathlib import Path

KIM_SUFFIX = "__MO_000000000000_000"
TS_FORMAT = "%Y%m%d_%H%M%S"
MAX_STEM_LEN = 40


def make_timestamp() -> str:
    return datetime.now().strftime(TS_FORMAT)


def make_model_stem(
    model_name: str | None,
    dataset_name: str | None,
    elements: list[str] | None,
) -> str:
    if model_name is not None:
        raw = model_name
    elif dataset_name is not None:
        raw = dataset_name
    elif elements:
        raw = "_".join(sorted(elements))
    else:
        raw = "colabfit_mace"
    return raw[:MAX_STEM_LEN]


def model_dir_name(stem: str, timestamp: str) -> str:
    return f"{stem}_{timestamp}"


def kim_model_dir_name(model_dir_stem: str) -> str:
    return f"{model_dir_stem}{KIM_SUFFIX}"


def extract_model_id(model_path) -> str:
    return Path(model_path).parent.name


def inference_file_name(model_id: str, structure_tag: str, timestamp: str) -> str:
    return f"{model_id}__{structure_tag}_{timestamp}.extxyz"


def test_driver_dir_name(model_id: str, driver_name: str, timestamp: str) -> str:
    return f"{model_id}__{driver_name}_{timestamp}"


def training_log_name(timestamp: str) -> str:
    return f"training_{timestamp}.log"


def structure_file_name(
    formula: str,
    crystal_structure: str | None,
    repeat: list[int] | None,
    timestamp: str,
) -> str:
    struct_tag = (crystal_structure or "molecule").lower()
    repeat_tag = f"_{'x'.join(str(r) for r in repeat)}" if repeat else ""
    return f"{formula}_{struct_tag}{repeat_tag}_{timestamp}.extxyz"
