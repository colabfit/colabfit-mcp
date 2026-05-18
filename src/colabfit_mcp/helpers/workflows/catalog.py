from colabfit_mcp.helpers.workflows.nodes import (
    make_build_dataset_node,
    make_check_local_node,
    make_create_structure_node,
    make_download_node,
    make_list_test_drivers_node,
    make_run_test_driver_node,
    make_search_node,
    make_select_dataset_node,
    make_train_mace_node,
    make_use_model_node,
    NodeFn,
)


NODE_REGISTRY: dict = {
    "check_local": make_check_local_node,
    "search": make_search_node,
    "select_dataset": make_select_dataset_node,
    "download": make_download_node,
    "build_dataset": make_build_dataset_node,
    "train_mace": make_train_mace_node,
    "create_structure": make_create_structure_node,
    "use_model": make_use_model_node,
    "list_test_drivers": make_list_test_drivers_node,
    "run_test_driver": make_run_test_driver_node,
}


NODE_CATALOG: dict[str, dict] = {
    "check_local": {
        "description": "Scan local dataset directory for previously downloaded datasets.",
        "params": ["elements", "property_types"],
        "reads": ["elements", "property_types"],
        "writes": ["local_matches", "local_dataset_match"],
    },
    "search": {
        "description": "Search ColabFit for datasets by text, elements, size.",
        "params": ["text", "elements", "property_types", "min_configurations",
                   "max_configurations", "exact_elements", "sort_by",
                   "sort_direction", "page", "page_size"],
        "reads": ["query", "elements", "property_types",
                  "min_configurations", "max_configurations"],
        "writes": ["search_results"],
    },
    "select_dataset": {
        "description": "Pick one dataset from search_results by strategy.",
        "params": ["strategy", "elements"],
        "strategies": ["first", "smallest", "largest"],
        "reads": ["search_results"],
        "writes": ["selected_dataset"],
    },
    "download": {
        "description": "Download a HuggingFace-cached ColabFit dataset.",
        "params": ["dataset_name", "dataset_id", "split", "n_configs"],
        "reads": ["selected_dataset", "dataset_name", "dataset_id"],
        "writes": ["dataset_name", "dataset_id", "dataset_path", "hf_id",
                   "dataset_analysis"],
    },
    "build_dataset": {
        "description": "Build a custom filtered dataset from ColabFit VastDB.",
        "params": ["methods_contain", "software_contain", "dataset_ids",
                   "formulae", "properties", "elements", "num_configs",
                   "dataset_name", "preview_only"],
        "reads": ["elements", "property_types", "dataset_name"],
        "writes": ["dataset_name", "dataset_path", "dataset_analysis"],
    },
    "train_mace": {
        "description": "Train a KLAY MACE-style model with KLIFF.",
        "params": ["train_file", "model_name", "dataset_name", "r_max",
                   "max_num_epochs", "batch_size", "device", "elements",
                   "n_layers", "avg_num_neighbors"],
        "reads": ["training_params", "dataset_name", "local_dataset_match"],
        "writes": ["model_path", "kim_model_name", "training_metrics"],
    },
    "create_structure": {
        "description": "Build an ASE Atoms structure and save as extxyz.",
        "params": ["formula", "crystal_structure", "lattice_constant", "repeat"],
        "reads": ["formula", "crystal_structure", "lattice_constant", "repeat"],
        "writes": ["structure_file"],
    },
    "use_model": {
        "description": "Run ASE calculations with a trained KLAY model.",
        "params": ["model_path", "formula", "crystal_structure",
                   "lattice_constant", "repeat", "structures", "input_file",
                   "calculations", "device", "mode"],
        "reads": ["model_path", "formula", "crystal_structure",
                  "lattice_constant", "repeat", "structures", "structure_file",
                  "calculations"],
        "writes": ["inference_results", "inference_output_file"],
    },
    "list_test_drivers": {
        "description": "List available kimvv test drivers.",
        "params": ["property_keyword"],
        "reads": [],
        "writes": ["selected_drivers"],
    },
    "run_test_driver": {
        "description": "Run a kimvv test driver against a trained model.",
        "params": ["driver", "model_path", "formula", "crystal_structure",
                   "lattice_constant", "device", "input_file", "structures",
                   "async_mode"],
        "reads": ["model_path", "formula", "crystal_structure",
                  "lattice_constant", "structures"],
        "writes": ["test_driver_results"],
    },
}


__all__ = ["NODE_REGISTRY", "NODE_CATALOG", "NodeFn"]
