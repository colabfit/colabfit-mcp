import operator
from typing import Annotated, TypedDict


class WorkflowState(TypedDict, total=False):
    workflow_id: str
    spec_name: str

    query: str
    elements: list[str]
    property_types: list[str]
    min_configurations: int
    max_configurations: int
    search_results: list[dict]
    selected_dataset: dict
    local_matches: list[dict]
    local_dataset_match: dict

    dataset_name: str
    dataset_id: str
    dataset_path: str
    dataset_ref: str
    hf_id: str
    safe_name: str
    split: str
    dataset_analysis: dict

    training_params: dict
    model_path: str
    model_dir: str
    kim_model_name: str
    training_metrics: dict
    training_log: str
    training_architecture: dict

    formula: str
    crystal_structure: str
    lattice_constant: float
    repeat: list[int]
    structures: list[dict]
    structure_file: str

    calculations: list[str]
    inference_results: list[dict]
    inference_output_file: str

    selected_drivers: list[str]
    test_driver_results: Annotated[dict, lambda a, b: {**(a or {}), **(b or {})}]

    node_history: Annotated[list[dict], operator.add]
    errors: Annotated[list[dict], operator.add]


JSON_SAFE_TYPES = (str, int, float, bool, list, dict, type(None))


def is_json_safe(value) -> bool:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(is_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_safe(v) for k, v in value.items())
    return False
