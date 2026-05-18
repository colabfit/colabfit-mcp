import json
from pathlib import Path

from pydantic import ValidationError

from colabfit_mcp.helpers.workflows.catalog import NODE_CATALOG
from colabfit_mcp.helpers.workflows.engine import (
    get_thread_state,
    resume_graph,
    run_graph,
)
from colabfit_mcp.helpers.workflows.routers import ROUTER_CATALOG
from colabfit_mcp.helpers.workflows.spec import GraphSpec


_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "helpers" / "workflows" / "templates"

_SPEC_RUNTIME: dict[str, GraphSpec] = {}


def _validate(spec: dict | str) -> tuple[GraphSpec | None, str | None]:
    if isinstance(spec, str):
        try:
            spec = json.loads(spec)
        except Exception as e:
            return None, f"spec is not valid JSON: {e}"
    try:
        return GraphSpec.model_validate(spec), None
    except ValidationError as e:
        return None, str(e)


def compile_workflow(spec: dict | str) -> dict:
    """Validate a graph spec without executing it.

    Returns a summary so the LLM (or a human) can preview the workflow before
    paying for a real run. Validates the same way run_workflow does: closed
    enums on node `type` and `router`, unique ids, all edge targets resolve.

    Args:
        spec: A graph spec object or JSON string.

    Returns:
        dict with success, plus name, n_nodes, n_edges, node_types, interrupts,
        and any long_running_nodes (train_mace, run_test_driver) so the caller
        can plan for polling.
    """
    parsed, err = _validate(spec)
    if err:
        return {"success": False, "error": f"spec validation failed: {err}"}
    long_running = {"train_mace", "run_test_driver", "build_dataset", "download"}
    long_running_nodes = [n.id for n in parsed.nodes if n.type in long_running]
    return {
        "success": True,
        "name": parsed.name,
        "n_nodes": len(parsed.nodes),
        "n_edges": len(parsed.edges),
        "node_types": sorted({n.type for n in parsed.nodes}),
        "interrupts_before": list(parsed.interrupts_before),
        "interrupts_after": list(parsed.interrupts_after),
        "long_running_nodes": long_running_nodes,
        "next_step": (
            "Spec is valid. Call run_workflow with the same spec to execute. "
            f"{len(long_running_nodes)} long-running node(s) — consider polling "
            "get_workflow_status." if long_running_nodes else
            "Spec is valid. Call run_workflow with the same spec to execute."
        ),
    }


def run_workflow(spec: dict | str, initial_state: dict | None = None,
                 workflow_id: str | None = None) -> dict:
    """Compile and execute a LangGraph workflow spec server-side.

    Workflows let an LLM describe a multi-step ColabFit operation (search →
    download → train → evaluate) as a graph rather than as a chain of imperative
    tool calls. The MCP server validates the spec, compiles it to a LangGraph
    StateGraph, and runs it to completion with SQLite checkpointing.

    Use a template via list_workflow_templates / get_workflow_template when one
    fits — only author a custom spec when no template matches.

    Args:
        spec: A graph spec object or JSON string. See list_workflow_nodes() for
            the available node types, list_workflow_routers() for branch
            predicates.
        initial_state: Optional dict merged into the spec's initial_state at
            invoke time (overrides per-run).
        workflow_id: Optional thread id to bind to; useful for re-running a
            named workflow. A random id is generated if omitted.

    Returns:
        dict with workflow_id, state (final state after run), interrupted (bool).
        If interrupted, call resume_workflow with the same workflow_id.
    """
    parsed, err = _validate(spec)
    if err:
        return {"success": False, "error": f"spec validation failed: {err}"}
    try:
        result = run_graph(parsed, initial_state=initial_state, workflow_id=workflow_id)
    except Exception as e:
        return {"success": False, "error": f"execution failed: {e}"}
    _SPEC_RUNTIME[result["workflow_id"]] = parsed
    return {"success": True, **result}


def resume_workflow(workflow_id: str, state_patch: dict | None = None) -> dict:
    """Resume a previously-interrupted workflow.

    Args:
        workflow_id: id returned by run_workflow.
        state_patch: Optional dict merged into the checkpointed state before
            resuming (e.g. to override an auto-selected dataset).

    Returns:
        Same shape as run_workflow.
    """
    spec = _SPEC_RUNTIME.get(workflow_id)
    if spec is None:
        return {"success": False, "error": f"unknown workflow_id {workflow_id!r}; "
                "the MCP server may have restarted — re-submit the spec via run_workflow"}
    try:
        result = resume_graph(spec, workflow_id, state_patch=state_patch)
    except Exception as e:
        return {"success": False, "error": f"resume failed: {e}"}
    return {"success": True, **result}


def get_workflow_status(workflow_id: str) -> dict:
    """Get the current checkpointed state of a workflow.

    Args:
        workflow_id: id returned by run_workflow.

    Returns:
        dict with workflow_id, state, next_nodes (which nodes will run on
        resume), interrupted.
    """
    spec = _SPEC_RUNTIME.get(workflow_id)
    if spec is None:
        return {"success": False, "error": f"unknown workflow_id {workflow_id!r}"}
    try:
        return {"success": True, **get_thread_state(spec, workflow_id)}
    except Exception as e:
        return {"success": False, "error": f"status fetch failed: {e}"}


def list_workflow_nodes() -> dict:
    """List all node types available for workflow specs.

    Each entry describes the node's purpose, accepted params, state keys it
    reads, and state keys it writes. Pass one of these names as the `type`
    field of a node in a graph spec.

    Returns:
        dict with `nodes` (the catalog) and `routers` (predicate catalog).
    """
    return {
        "success": True,
        "nodes": NODE_CATALOG,
        "routers": ROUTER_CATALOG,
        "sentinels": {"start": "__start__", "end": "__end__"},
        "next_step": (
            "Author a spec using these node types and routers, or call "
            "list_workflow_templates to fetch a ready-made one."
        ),
    }


def list_workflow_templates() -> dict:
    """List bundled workflow spec templates.

    Templates are pre-authored graphs for common pipelines. Use
    get_workflow_template(name) to fetch one, override fields under
    initial_state, then pass it to run_workflow.

    Returns:
        dict with `templates`: list of {name, description, file}.
    """
    if not _TEMPLATES_DIR.exists():
        return {"success": True, "templates": []}
    out = []
    for p in sorted(_TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        out.append({
            "name": data.get("name", p.stem),
            "description": data.get("description", ""),
            "file": p.name,
        })
    return {"success": True, "templates": out}


def get_workflow_template(name: str) -> dict:
    """Fetch a bundled workflow template by name.

    Args:
        name: The template `name` (or filename stem) as returned by
            list_workflow_templates.

    Returns:
        dict with the full spec under `spec`, ready to customize and pass to
        run_workflow.
    """
    if not _TEMPLATES_DIR.exists():
        return {"success": False, "error": "templates directory not found"}
    candidates = [
        _TEMPLATES_DIR / f"{name}.json",
        _TEMPLATES_DIR / name,
    ]
    for p in candidates:
        if p.exists():
            try:
                spec = json.loads(p.read_text())
            except Exception as e:
                return {"success": False, "error": f"template parse error: {e}"}
            return {"success": True, "spec": spec, "file": p.name}
    return {"success": False, "error": f"template {name!r} not found"}
