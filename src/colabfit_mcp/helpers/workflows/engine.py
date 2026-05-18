import threading
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from colabfit_mcp.config import DATA_ROOT
from colabfit_mcp.helpers.workflows.catalog import NODE_REGISTRY
from colabfit_mcp.helpers.workflows.routers import ROUTER_REGISTRY
from colabfit_mcp.helpers.workflows.spec import EdgeSpec, GraphSpec, NodeSpec, START_SENTINEL, END_SENTINEL
from colabfit_mcp.helpers.workflows.state import WorkflowState


_WORKFLOWS_DIR = DATA_ROOT / "workflows"
_WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)
_SQLITE_PATH = _WORKFLOWS_DIR / "checkpoints.sqlite"

_train_lock = threading.Lock()

_SAVER_CTX = None
_SAVER = None


def get_saver():
    global _SAVER, _SAVER_CTX
    if _SAVER is None:
        _SAVER_CTX = SqliteSaver.from_conn_string(str(_SQLITE_PATH))
        _SAVER = _SAVER_CTX.__enter__()
    return _SAVER


def reset_saver_for_test():
    global _SAVER, _SAVER_CTX
    if _SAVER_CTX is not None:
        _SAVER_CTX.__exit__(None, None, None)
    _SAVER_CTX = None
    _SAVER = MemorySaver()
    return _SAVER


def _wrap_train_lock(fn):
    def wrapped(state: dict) -> dict:
        with _train_lock:
            return fn(state)
    return wrapped


def _build_node(node: NodeSpec):
    factory = NODE_REGISTRY[node.type]
    fn = factory(node.params)
    if node.type == "train_mace":
        fn = _wrap_train_lock(fn)
    return fn


def _resolve(name: str) -> str:
    if name == START_SENTINEL:
        return START
    if name == END_SENTINEL:
        return END
    return name


def compile_spec(spec: GraphSpec, saver=None):
    g = StateGraph(WorkflowState)
    for node in spec.nodes:
        g.add_node(node.id, _build_node(node))

    for edge in spec.edges:
        _add_edge(g, edge)

    compile_kwargs = {}
    if saver is None:
        saver = get_saver()
    if saver is not None:
        compile_kwargs["checkpointer"] = saver
    if spec.interrupts_before:
        compile_kwargs["interrupt_before"] = spec.interrupts_before
    if spec.interrupts_after:
        compile_kwargs["interrupt_after"] = spec.interrupts_after
    return g.compile(**compile_kwargs)


def _add_edge(g: StateGraph, edge: EdgeSpec) -> None:
    src = _resolve(edge.source)
    if edge.target is not None:
        g.add_edge(src, _resolve(edge.target))
        return
    if edge.fan_out is not None:
        for tgt in edge.fan_out:
            g.add_edge(src, _resolve(tgt))
        return
    if edge.router is not None and edge.branches is not None:
        router_fn = ROUTER_REGISTRY[edge.router]
        mapping = {key: _resolve(val) for key, val in edge.branches.items()}
        g.add_conditional_edges(src, router_fn, mapping)


def new_workflow_id() -> str:
    return uuid.uuid4().hex


def run_graph(spec: GraphSpec, initial_state: dict | None = None,
              workflow_id: str | None = None, saver=None) -> dict:
    compiled = compile_spec(spec, saver=saver)
    wid = workflow_id or new_workflow_id()
    config = {"configurable": {"thread_id": wid}}
    state: dict = dict(spec.initial_state)
    if initial_state:
        state.update(initial_state)
    state["workflow_id"] = wid
    state["spec_name"] = spec.name
    final = compiled.invoke(state, config=config)
    return {"workflow_id": wid, "state": _serialize_state(final),
            "interrupted": _is_interrupted(final)}


def resume_graph(spec: GraphSpec, workflow_id: str,
                 state_patch: dict | None = None, saver=None) -> dict:
    compiled = compile_spec(spec, saver=saver)
    config = {"configurable": {"thread_id": workflow_id}}
    if state_patch:
        compiled.update_state(config, state_patch)
    final = compiled.invoke(None, config=config)
    return {"workflow_id": workflow_id, "state": _serialize_state(final),
            "interrupted": _is_interrupted(final)}


def get_thread_state(spec: GraphSpec, workflow_id: str, saver=None) -> dict:
    compiled = compile_spec(spec, saver=saver)
    config = {"configurable": {"thread_id": workflow_id}}
    snap = compiled.get_state(config)
    return {
        "workflow_id": workflow_id,
        "state": _serialize_state(snap.values if snap else {}),
        "next_nodes": list(snap.next) if snap else [],
        "interrupted": bool(snap and snap.next),
    }


def _is_interrupted(final: dict) -> bool:
    return False


def _serialize_state(state: dict) -> dict:
    out: dict = {}
    for k, v in (state or {}).items():
        try:
            import json
            json.dumps(v, default=str)
            out[k] = v
        except Exception:
            out[k] = str(v)
    return out
