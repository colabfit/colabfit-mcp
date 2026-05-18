from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

from colabfit_mcp.helpers.workflows.catalog import NODE_REGISTRY
from colabfit_mcp.helpers.workflows.routers import ROUTER_REGISTRY


START_SENTINEL = "__start__"
END_SENTINEL = "__end__"


class NodeSpec(BaseModel):
    id: str
    type: str
    params: dict = Field(default_factory=dict)
    on_error: Literal["fail", "continue", "retry"] = "fail"
    retries: int = 0

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in NODE_REGISTRY:
            raise ValueError(
                f"unknown node type {v!r}; valid: {sorted(NODE_REGISTRY.keys())}"
            )
        return v

    @field_validator("id")
    @classmethod
    def _id_not_sentinel(cls, v: str) -> str:
        if v in (START_SENTINEL, END_SENTINEL):
            raise ValueError(f"node id may not be {v!r}")
        return v


class EdgeSpec(BaseModel):
    source: str = Field(alias="from")
    target: str | None = Field(default=None, alias="to")
    router: str | None = None
    branches: dict[str, str] | None = None
    fan_out: list[str] | None = None

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _exactly_one_kind(self):
        kinds = sum(
            x is not None
            for x in (self.target, self.router, self.fan_out)
        )
        if kinds != 1:
            raise ValueError(
                "each edge must have exactly one of: to, router, fan_out"
            )
        if self.router is not None:
            if self.router not in ROUTER_REGISTRY:
                raise ValueError(
                    f"unknown router {self.router!r}; valid: "
                    f"{sorted(ROUTER_REGISTRY.keys())}"
                )
            if not self.branches:
                raise ValueError("router edges require non-empty branches")
        return self


class GraphSpec(BaseModel):
    spec_version: str = "1"
    name: str
    description: str = ""
    initial_state: dict = Field(default_factory=dict)
    nodes: list[NodeSpec]
    edges: list[EdgeSpec]
    interrupts_before: list[str] = Field(default_factory=list)
    interrupts_after: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_unique_and_edges_valid(self):
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate node ids")
        known = set(ids) | {START_SENTINEL, END_SENTINEL}
        for e in self.edges:
            if e.source not in known:
                raise ValueError(f"edge source {e.source!r} is not a known node")
            targets: list[str] = []
            if e.target is not None:
                targets.append(e.target)
            if e.fan_out is not None:
                targets.extend(e.fan_out)
            if e.branches is not None:
                targets.extend(e.branches.values())
            for t in targets:
                if t not in known:
                    raise ValueError(f"edge target {t!r} is not a known node")
        for nid in self.interrupts_before + self.interrupts_after:
            if nid not in set(ids):
                raise ValueError(f"interrupt refers to unknown node id {nid!r}")
        if not any(e.source == START_SENTINEL for e in self.edges):
            raise ValueError("graph must have at least one edge from __start__")
        return self
