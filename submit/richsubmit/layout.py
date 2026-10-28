"""Placement of MPI ranks on nodes: ranks per node, node count, memory and GPUs per rank."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from .slurm import Partition


@dataclass
class Resources:
    """What the user asks for. Zero means "not set" for the optional fields."""
    ntasks: int = 1
    cores_per_task: int = 1
    ranks_per_node: int = 0          # 0: as many as fit, spread evenly over the nodes
    min_memory_per_rank_mb: int = 0  # 0: no memory constraint
    gpus_per_rank: int = 0           # GPUs each rank sees; 0 when no feature runs on a GPU
    ranks_per_gpu: int = 1           # >1 oversubscribes: that many ranks share one GPU

    @classmethod
    def from_dict(cls, data: Dict) -> "Resources":
        known = {key: int(value) for key, value in (data or {}).items()
                 if key in cls.__dataclass_fields__ and value not in (None, "")}
        return cls(**known)


@dataclass
class Layout:
    nodes: int = 0
    ranks_per_node: int = 0
    max_ranks_per_node: int = 0
    limited_by: str = ""               # what bounds max_ranks_per_node
    memory_per_rank_mb: int = 0        # node memory split over its ranks
    memory_per_node_mb: int = 0        # what is requested from SLURM (0: whole-node default)
    gpus_per_node: int = 0             # GPUs the ranks of one node use
    idle_cores_per_node: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def compute(resources: Resources, partition: Partition, uses_gpus: bool) -> Layout:
    """Fit the requested ranks onto the partition's nodes.

    A rank lives on one node, so its cores, memory and GPUs come from that node:
    cores_per_task <= cores per node, gpus_per_rank <= GPUs per node.
    """
    layout = Layout()
    r = resources
    if r.ntasks < 1:
        layout.errors.append("The number of tasks must be at least 1.")
    if r.cores_per_task < 1:
        layout.errors.append("Cores per task must be at least 1.")
    if r.ranks_per_node < 0 or r.min_memory_per_rank_mb < 0 or r.gpus_per_rank < 0 or r.ranks_per_gpu < 1:
        layout.errors.append("Ranks per node, memory per rank and GPUs per rank cannot be negative; "
                             "ranks per GPU must be at least 1.")
    if layout.errors:
        return layout

    cores = partition.cores_per_node
    if cores and r.cores_per_task > cores:
        layout.errors.append(f"{r.cores_per_task} cores per task exceed the {cores} cores of a "
                             f"{partition.name} node.")

    gpus_per_rank, ranks_per_gpu = r.gpus_per_rank, r.ranks_per_gpu
    if uses_gpus:
        if not partition.has_gpus:
            layout.errors.append(f"A feature runs on a GPU, but partition {partition.name} has no GPUs.")
        if gpus_per_rank < 1:
            gpus_per_rank = 1
        if gpus_per_rank > 1 and ranks_per_gpu > 1:
            layout.errors.append("Ranks can share a GPU only when each rank uses one GPU "
                                 "(set GPUs per rank to 1, or ranks per GPU to 1).")
        if partition.has_gpus and gpus_per_rank > partition.gpus_per_node:
            layout.errors.append(f"{gpus_per_rank} GPUs per rank exceed the {partition.gpus_per_node} "
                                 f"GPUs of a {partition.name} node (a rank cannot use another node's GPUs).")
    else:
        gpus_per_rank, ranks_per_gpu = 0, 1
    if layout.errors:
        return layout

    limits = []
    if cores:
        limits.append((cores // r.cores_per_task, f"cores ({cores} per node / {r.cores_per_task} per task)"))
    if r.min_memory_per_rank_mb and partition.memory_per_node_mb:
        limits.append((partition.memory_per_node_mb // r.min_memory_per_rank_mb,
                       f"memory ({partition.memory_per_node_mb} MiB per node / "
                       f"{r.min_memory_per_rank_mb} MiB per rank)"))
    if uses_gpus and partition.has_gpus:
        if ranks_per_gpu > 1:
            limits.append((partition.gpus_per_node * ranks_per_gpu,
                           f"GPUs ({partition.gpus_per_node} per node x {ranks_per_gpu} ranks per GPU)"))
        else:
            limits.append((partition.gpus_per_node // gpus_per_rank,
                           f"GPUs ({partition.gpus_per_node} per node / {gpus_per_rank} per rank)"))
    if limits:
        layout.max_ranks_per_node, layout.limited_by = min(limits, key=lambda item: item[0])
    else:
        layout.max_ranks_per_node, layout.limited_by = r.ntasks, "nothing (no node resources known)"
    if layout.max_ranks_per_node < 1:
        layout.errors.append(f"Not even one rank fits a {partition.name} node; limited by {layout.limited_by}.")
        return layout

    if r.ranks_per_node:
        if r.ranks_per_node > layout.max_ranks_per_node:
            layout.errors.append(f"{r.ranks_per_node} ranks per node exceed the {layout.max_ranks_per_node} "
                                 f"that fit, limited by {layout.limited_by}.")
            return layout
        layout.ranks_per_node = min(r.ranks_per_node, r.ntasks)
        layout.nodes = _ceil_div(r.ntasks, layout.ranks_per_node)
    else:
        layout.nodes = _ceil_div(r.ntasks, layout.max_ranks_per_node)
        layout.ranks_per_node = _ceil_div(r.ntasks, layout.nodes)
    if r.ntasks % layout.ranks_per_node:
        layout.warnings.append(f"{r.ntasks} tasks do not fill {layout.nodes} nodes of {layout.ranks_per_node} "
                               f"ranks evenly; the last node runs {r.ntasks % layout.ranks_per_node}.")
    if partition.nodes and layout.nodes > partition.nodes:
        layout.errors.append(f"{layout.nodes} nodes exceed the {partition.nodes} of partition {partition.name}.")

    if partition.memory_per_node_mb:
        layout.memory_per_rank_mb = partition.memory_per_node_mb // layout.ranks_per_node
    if r.min_memory_per_rank_mb:
        layout.memory_per_node_mb = r.min_memory_per_rank_mb * layout.ranks_per_node
    if cores:
        layout.idle_cores_per_node = cores - layout.ranks_per_node * r.cores_per_task
    if uses_gpus:
        layout.gpus_per_node = (_ceil_div(layout.ranks_per_node, ranks_per_gpu) if ranks_per_gpu > 1
                                else layout.ranks_per_node * gpus_per_rank)
        if ranks_per_gpu > 1:
            layout.warnings.append(f"GPU oversubscription: {ranks_per_gpu} ranks share each GPU.")
    return layout
