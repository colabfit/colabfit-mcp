from typing import Callable


RouterFn = Callable[[dict], str]


def has_local_match(state: dict) -> str:
    return "yes" if state.get("local_dataset_match") else "no"


def has_search_results(state: dict) -> str:
    return "yes" if state.get("search_results") else "no"


def training_succeeded(state: dict) -> str:
    return "yes" if state.get("model_path") else "no"


def has_errors(state: dict) -> str:
    return "yes" if state.get("errors") else "no"


def has_selected_dataset(state: dict) -> str:
    return "yes" if state.get("selected_dataset") else "no"


def has_model(state: dict) -> str:
    return "yes" if state.get("model_path") else "no"


ROUTER_REGISTRY: dict[str, RouterFn] = {
    "has_local_match": has_local_match,
    "has_search_results": has_search_results,
    "training_succeeded": training_succeeded,
    "has_errors": has_errors,
    "has_selected_dataset": has_selected_dataset,
    "has_model": has_model,
}


ROUTER_CATALOG: dict[str, dict] = {
    "has_local_match": {
        "description": "Branch on whether check_local found a local dataset.",
        "returns": ["yes", "no"],
        "reads": ["local_dataset_match"],
    },
    "has_search_results": {
        "description": "Branch on whether search returned any results.",
        "returns": ["yes", "no"],
        "reads": ["search_results"],
    },
    "training_succeeded": {
        "description": "Branch on whether train_mace produced a model_path.",
        "returns": ["yes", "no"],
        "reads": ["model_path"],
    },
    "has_errors": {
        "description": "Branch on whether any node has recorded an error.",
        "returns": ["yes", "no"],
        "reads": ["errors"],
    },
    "has_selected_dataset": {
        "description": "Branch on whether a dataset is in state ready to download/train.",
        "returns": ["yes", "no"],
        "reads": ["selected_dataset"],
    },
    "has_model": {
        "description": "Branch on whether a model_path is in state.",
        "returns": ["yes", "no"],
        "reads": ["model_path"],
    },
}
