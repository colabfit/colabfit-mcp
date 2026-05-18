from typing import Callable

from colabfit_mcp.tools.search import search_datasets
from colabfit_mcp.tools.local_datasets import check_local_datasets
from colabfit_mcp.tools.download import download_dataset


NodeFn = Callable[[dict], dict]


def _record(name: str, status: str, **extra) -> dict:
    entry = {"node": name, "status": status}
    entry.update(extra)
    return entry


def _error(node: str, message: str) -> dict:
    return {"errors": [{"node": node, "error": message}]}


def make_check_local_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        result = check_local_datasets(
            elements=params.get("elements") or state.get("elements"),
            property_types=params.get("property_types") or state.get("property_types"),
        )
        if not result.get("success"):
            return _error("check_local", result.get("error", "unknown"))
        matches = result.get("matches", [])
        return {
            "local_matches": matches,
            "local_dataset_match": matches[0] if matches else None,
            "node_history": [_record("check_local", "ok", n_matches=len(matches))],
        }

    return node


def make_search_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        result = search_datasets(
            text=params.get("text") or state.get("query"),
            elements=params.get("elements") or state.get("elements"),
            property_types=params.get("property_types") or state.get("property_types"),
            min_configurations=params.get("min_configurations") or state.get("min_configurations"),
            max_configurations=params.get("max_configurations") or state.get("max_configurations"),
            exact_elements=bool(params.get("exact_elements", False)),
            sort_by=params.get("sort_by", "downloads"),
            sort_direction=params.get("sort_direction", "descending"),
            page=int(params.get("page", 1)),
            page_size=int(params.get("page_size", 10)),
        )
        if not result.get("success"):
            return _error("search", result.get("error", "unknown"))
        results = result.get("results", [])
        return {
            "search_results": results,
            "node_history": [_record("search", "ok", count=len(results))],
        }

    return node


def make_select_dataset_node(params: dict | None = None) -> NodeFn:
    params = params or {}
    strategy = params.get("strategy", "first")
    required_elements = params.get("elements")

    def node(state: dict) -> dict:
        candidates = state.get("search_results") or []
        if not candidates:
            return _error("select_dataset", "no search_results in state")
        if required_elements:
            req = {e.capitalize() for e in required_elements}
            candidates = [
                c for c in candidates
                if req.issubset({e.capitalize() for e in (c.get("elements") or [])})
            ]
        if not candidates:
            return _error("select_dataset", "no candidate matches required elements")
        if strategy == "smallest":
            pick = min(candidates, key=lambda c: c.get("nconfigurations", 1 << 31))
        elif strategy == "largest":
            pick = max(candidates, key=lambda c: c.get("nconfigurations", -1))
        else:
            pick = candidates[0]
        return {
            "selected_dataset": pick,
            "node_history": [_record("select_dataset", "ok", picked=pick.get("name"))],
        }

    return node


def make_download_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        selected = state.get("selected_dataset") or {}
        name = params.get("dataset_name") or selected.get("name") or state.get("dataset_name")
        ds_id = params.get("dataset_id") or selected.get("id") or state.get("dataset_id")
        if not name:
            return _error("download", "dataset_name not in state or params")
        result = download_dataset(
            dataset_name=name,
            dataset_id=ds_id,
            split=params.get("split", "train"),
            n_configs=params.get("n_configs"),
        )
        if not result.get("success"):
            return _error("download", result.get("error", "unknown"))
        return {
            "dataset_name": result.get("dataset_name"),
            "dataset_id": result.get("dataset_id"),
            "dataset_path": result.get("output_dir"),
            "dataset_ref": result.get("dataset_ref"),
            "hf_id": result.get("hf_id"),
            "dataset_analysis": result.get("analysis", {}),
            "node_history": [_record("download", "ok", cached=result.get("cached", False))],
        }

    return node


def make_build_dataset_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.build_dataset import build_dataset
        result = build_dataset(
            methods_contain=params.get("methods_contain"),
            software_contain=params.get("software_contain"),
            dataset_ids=params.get("dataset_ids"),
            formulae=params.get("formulae"),
            properties=params.get("properties") or state.get("property_types"),
            elements=params.get("elements") or state.get("elements"),
            num_configs=int(params.get("num_configs", 1000)),
            dataset_name=params.get("dataset_name") or state.get("dataset_name"),
            preview_only=bool(params.get("preview_only", False)),
        )
        if not result.get("success"):
            return _error("build_dataset", result.get("error", "unknown"))
        return {
            "dataset_name": result.get("dataset_name"),
            "dataset_path": result.get("output_dir") or result.get("dataset_path"),
            "dataset_ref": result.get("dataset_ref"),
            "dataset_analysis": result.get("analysis", {}),
            "node_history": [_record("build_dataset", "ok")],
        }

    return node


def make_train_mace_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.train import train_mace
        tp = {**(state.get("training_params") or {}), **params}
        local = state.get("local_dataset_match") or {}
        dataset_name = (
            tp.get("dataset_name")
            or state.get("dataset_name")
            or local.get("dataset_dir")
        )
        result = train_mace(
            train_file=tp.get("train_file"),
            model_name=tp.get("model_name"),
            dataset_name=dataset_name,
            r_max=float(tp.get("r_max", 5.0)),
            max_num_epochs=int(tp.get("max_num_epochs", 100)),
            batch_size=tp.get("batch_size"),
            device=tp.get("device"),
            elements=tp.get("elements") or state.get("elements"),
            n_layers=int(tp.get("n_layers", 2)),
            avg_num_neighbors=tp.get("avg_num_neighbors"),
        )
        if not result.get("success"):
            return _error("train_mace", result.get("error", "unknown"))
        return {
            "model_path": result.get("model_path_docker") or result.get("model_path"),
            "model_dir": result.get("model_dir"),
            "kim_model_name": result.get("kim_model_name"),
            "training_metrics": result.get("metrics", {}),
            "training_log": result.get("training_log_docker") or result.get("training_log"),
            "training_architecture": result.get("architecture", {}),
            "node_history": [_record("train_mace", "ok")],
        }

    return node


def make_create_structure_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.create_structure import create_structure
        result = create_structure(
            formula=params.get("formula") or state.get("formula"),
            crystal_structure=params.get("crystal_structure") or state.get("crystal_structure"),
            lattice_constant=params.get("lattice_constant") or state.get("lattice_constant"),
            repeat=params.get("repeat") or state.get("repeat"),
        )
        if not result.get("success"):
            return _error("create_structure", result.get("error", "unknown"))
        return {
            "structure_file": result.get("output_file"),
            "node_history": [_record("create_structure", "ok", n_atoms=result.get("n_atoms"))],
        }

    return node


def make_use_model_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.use_model import use_model
        result = use_model(
            model_path=params.get("model_path") or state.get("model_path"),
            formula=params.get("formula") or state.get("formula"),
            crystal_structure=params.get("crystal_structure") or state.get("crystal_structure"),
            lattice_constant=params.get("lattice_constant") or state.get("lattice_constant"),
            repeat=params.get("repeat") or state.get("repeat"),
            structures=params.get("structures") or state.get("structures"),
            input_file=params.get("input_file") or state.get("structure_file"),
            calculations=params.get("calculations") or state.get("calculations"),
            device=params.get("device"),
            mode=params.get("mode", "run"),
        )
        if not result.get("success"):
            return _error("use_model", result.get("error", "unknown"))
        return {
            "inference_results": result.get("frames", []),
            "inference_output_file": result.get("output_file"),
            "node_history": [_record("use_model", "ok")],
        }

    return node


def make_list_test_drivers_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.test_driver import list_test_drivers
        result = list_test_drivers(property_keyword=params.get("property_keyword"))
        if not result.get("success"):
            return _error("list_test_drivers", result.get("error", "unknown"))
        return {
            "selected_drivers": [d["name"] for d in result.get("test_drivers", [])],
            "node_history": [_record("list_test_drivers", "ok",
                                     count=result.get("total", 0))],
        }

    return node


def make_run_test_driver_node(params: dict | None = None) -> NodeFn:
    params = params or {}

    def node(state: dict) -> dict:
        from colabfit_mcp.tools.test_driver import run_test_driver
        driver_name = params.get("driver") or params.get("test_driver_name")
        if not driver_name:
            return _error("run_test_driver", "driver name not provided in params")
        result = run_test_driver(
            model_path=params.get("model_path") or state.get("model_path"),
            test_driver_name=driver_name,
            formula=params.get("formula") or state.get("formula"),
            crystal_structure=params.get("crystal_structure") or state.get("crystal_structure"),
            lattice_constant=params.get("lattice_constant") or state.get("lattice_constant"),
            device=params.get("device"),
            input_file=params.get("input_file"),
            structures=params.get("structures") or state.get("structures"),
            async_mode=bool(params.get("async_mode", False)),
        )
        if not result.get("success"):
            return _error(f"run_test_driver:{driver_name}", result.get("error", "unknown"))
        return {
            "test_driver_results": {driver_name: result.get("results", [])},
            "node_history": [_record(f"run_test_driver:{driver_name}", "ok")],
        }

    return node
