"""SLURM queries: partitions, their nodes, accounts, and job submission."""

from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple


class SlurmError(RuntimeError):
    pass


@dataclass
class Partition:
    """One partition and the resources of its smallest node.

    Nodes of one partition can differ; every limit below is the minimum over
    the partition's nodes, so a layout that fits it fits any node.
    """
    name: str
    state: str
    default: bool
    nodes: int
    max_time: str                 # SLURM text, e.g. 7-00:00:00 or UNLIMITED
    default_time: str
    allowed_qos: List[str]
    cpus_per_node: int            # CPUs SLURM hands out (CPUEfctv: core specialization removed)
    threads_per_core: int
    cores_per_node: int           # cpus_per_node / threads_per_core
    memory_per_node_mb: int       # RealMemory
    gpus_per_node: int
    gpu_type: str = ""
    features: List[str] = field(default_factory=list)

    @property
    def has_gpus(self) -> bool:
        return self.gpus_per_node > 0

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["has_gpus"] = self.has_gpus
        data["max_time_minutes"] = parse_time_minutes(self.max_time)
        return data


def run(command: List[str]) -> str:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True)
    except FileNotFoundError as error:
        raise SlurmError(f"{command[0]} is not available: {error}") from error
    if result.returncode != 0:
        raise SlurmError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result.stdout


def parse_memory_mb(value: str) -> int:
    """SLURM memory text (512000, 500G, 2.5T) in MiB; plain numbers are MiB."""
    text = str(value).strip().rstrip("+")
    if not text:
        return 0
    multiplier = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}.get(text[-1].upper())
    if multiplier is None:
        multiplier = 1
    else:
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def parse_time_minutes(value: str) -> Optional[int]:
    """SLURM time text in minutes; None for UNLIMITED or unparsable text.

    Accepted: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.
    """
    text = str(value).strip()
    if not text or text.upper() in {"UNLIMITED", "INFINITE", "N/A", "NONE"}:
        return None
    days = 0
    if "-" in text:
        day_text, text = text.split("-", 1)
        if not day_text.isdigit():
            return None
        days = int(day_text)
        parts = text.split(":")
        if not all(part.isdigit() for part in parts) or len(parts) > 3:
            return None
        hours, minutes, seconds = (list(map(int, parts)) + [0, 0])[:3]
    else:
        parts = text.split(":")
        if not all(part.isdigit() for part in parts) or len(parts) > 3:
            return None
        numbers = list(map(int, parts))
        if len(numbers) == 1:
            hours, minutes, seconds = 0, numbers[0], 0
        elif len(numbers) == 2:
            hours, minutes, seconds = 0, numbers[0], numbers[1]
        else:
            hours, minutes, seconds = numbers
    return days * 24 * 60 + hours * 60 + minutes + (1 if seconds else 0)


def format_time(minutes: int) -> str:
    days, rest = divmod(int(minutes), 24 * 60)
    hours, mins = divmod(rest, 60)
    return f"{days}-{hours:02d}:{mins:02d}:00" if days else f"{hours:02d}:{mins:02d}:00"


def parse_gres_gpus(gres: str) -> Tuple[int, str]:
    """GPU count and type from a Gres string such as gpu:8(S:0-7) or gpu:a100:4,shard:8."""
    total, kind = 0, ""
    for item in gres.split(","):
        match = re.match(r"gpu(?::([A-Za-z][\w.-]*))?:(\d+)", item.strip())
        if match:
            total += int(match.group(2))
            kind = kind or (match.group(1) or "")
    return total, kind


def _field(record: str, name: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(name)}=(\S*)", record)
    return match.group(1) if match else ""


def _int(text: str, default: int = 0) -> int:
    try:
        return int(text)
    except (TypeError, ValueError):
        return default


def _show(kind: str, name: Optional[str] = None) -> List[str]:
    command = ["scontrol", "show", kind] + ([name] if name else []) + ["--oneliner"]
    return [line for line in run(command).splitlines() if line.strip()]


def _node_resources(node_records: List[str]) -> Dict:
    """Minimum resources over the given nodes."""
    nodes = []
    for record in node_records:
        threads = max(_int(_field(record, "ThreadsPerCore"), 1), 1)
        cpus = _int(_field(record, "CPUEfctv")) or _int(_field(record, "CPUTot"))
        gpus, gpu_type = parse_gres_gpus(_field(record, "Gres"))
        nodes.append({
            "cpus": cpus,
            "threads": threads,
            "memory": parse_memory_mb(_field(record, "RealMemory")),
            "gpus": gpus,
            "gpu_type": gpu_type,
            "features": [f for f in _field(record, "AvailableFeatures").split(",") if f and f != "(null)"],
        })
    if not nodes:
        return {}
    smallest_cpu = min(nodes, key=lambda node: node["cpus"] // node["threads"])
    return {
        "cpus_per_node": smallest_cpu["cpus"],
        "threads_per_core": smallest_cpu["threads"],
        "cores_per_node": smallest_cpu["cpus"] // smallest_cpu["threads"],
        "memory_per_node_mb": min(node["memory"] for node in nodes),
        "gpus_per_node": min(node["gpus"] for node in nodes),
        "gpu_type": next((node["gpu_type"] for node in nodes if node["gpu_type"]), ""),
        "features": sorted(set.intersection(*(set(node["features"]) for node in nodes))),
    }


def _sample_nodes(node_expression: str, limit: int) -> str:
    """At most `limit` node names of a hostlist expression, spread over it.

    Large machines have thousands of nodes per partition; querying a spread
    sample keeps `scontrol show node` fast and still sees heterogeneous ranges.
    """
    names = run(["scontrol", "show", "hostnames", node_expression]).split()
    if len(names) <= limit:
        return ",".join(names)
    step = len(names) / limit
    return ",".join(names[int(i * step)] for i in range(limit))


def partitions(sample_nodes: int = 32) -> List[Partition]:
    """All partitions visible to the user, with their node resources."""
    result = []
    for record in _show("partition"):
        name = _field(record, "PartitionName")
        if not name or _field(record, "Hidden").upper() == "YES":
            continue
        node_expression = _field(record, "Nodes")
        resources = {}
        if node_expression and node_expression != "(null)":
            resources = _node_resources(_show("node", _sample_nodes(node_expression, sample_nodes)))
        qos = [q for q in _field(record, "AllowQos").split(",") if q and q != "ALL"]
        result.append(Partition(
            name=name,
            state=_field(record, "State"),
            default=_field(record, "Default").upper() == "YES",
            nodes=_int(_field(record, "TotalNodes")),
            max_time=_field(record, "MaxTime"),
            default_time=_field(record, "DefaultTime"),
            allowed_qos=qos,
            cpus_per_node=resources.get("cpus_per_node", 0),
            threads_per_core=resources.get("threads_per_core", 1),
            cores_per_node=resources.get("cores_per_node", 0),
            memory_per_node_mb=resources.get("memory_per_node_mb", 0),
            gpus_per_node=resources.get("gpus_per_node", 0),
            gpu_type=resources.get("gpu_type", ""),
            features=resources.get("features", []),
        ))
    return result


def partition(name: str) -> Partition:
    for candidate in partitions():
        if candidate.name == name:
            return candidate
    raise SlurmError(f"No SLURM partition named {name!r}")


def accounts(user: Optional[str] = None) -> List[Dict[str, List[str]]]:
    """The user's accounts and the QOS each allows, from sacctmgr; empty when unavailable."""
    import getpass
    try:
        text = run(["sacctmgr", "-n", "-P", "show", "assoc", f"user={user or getpass.getuser()}",
                    "format=account,qos"])
    except SlurmError:
        return []
    found: Dict[str, set] = {}
    for line in text.splitlines():
        if "|" not in line:
            continue
        account, qos = line.split("|", 1)
        found.setdefault(account.strip(), set()).update(q for q in qos.strip().split(",") if q)
    return [{"name": name, "qos": sorted(qos)} for name, qos in sorted(found.items())]


def sbatch(script_path: str, extra_args: Optional[List[str]] = None, cwd: Optional[str] = None) -> str:
    """Submit a batch script; returns the job ID."""
    command = ["sbatch", "--parsable"] + list(extra_args or []) + [script_path]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, cwd=cwd)
    except FileNotFoundError as error:
        raise SlurmError(f"sbatch is not available: {error}") from error
    if result.returncode != 0:
        raise SlurmError(f"sbatch failed: {result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]
