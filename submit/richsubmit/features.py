"""Parallel features of RICH, STORM and MadVoro, and how a choice of backends resolves.

Each project describes its features in a JSON manifest (submit/features/*.json).
A feature offers backends out of serial, openmp and gpu; a subfeature can also
"inherit" its parent's. A backend says what it needs and what it sets:

  build      alternatives of build-variable values, e.g. [{"kokkos": "host"}, {"kokkos": "device"}]
  threads    "global": the rank's one OpenMP/Kokkos host pool (OMP_NUM_THREADS) runs it;
             "feature": it honours its own thread count (a program argument or the execution config)
  max_gpus   GPUs one rank of the feature can use
  args       program options set when the benchmark declares them, e.g. {"voronoi-subdomains": "{threads}"}
  env        environment variables for the run
  requires   backends other features must have, e.g. {"rich.hydro.slopes": ["gpu"]}
  available  false while the code cannot honour the backend yet; note says why
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BACKENDS = ("serial", "openmp", "gpu", "inherit")
BACKEND_LABEL = {"serial": "Serial", "openmp": "OpenMP", "gpu": "GPU", "inherit": "Same as parent"}
BACKEND_KEYS = {"build", "threads", "max_gpus", "args", "env", "requires", "available", "note"}


class ManifestError(ValueError):
    pass


@dataclass
class Feature:
    id: str
    title: str
    project: str
    description: str = ""
    default: str = "serial"
    backends: Dict[str, Dict] = field(default_factory=dict)
    parent: str = ""
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Catalog:
    projects: List[Dict] = field(default_factory=list)     # [{"id", "title", "features": [top-level ids]}]
    features: Dict[str, Feature] = field(default_factory=dict)  # parents before their children
    rules: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"projects": self.projects,
                "features": {fid: feature.to_dict() for fid, feature in self.features.items()},
                "rules": self.rules}


def _check_backend(feature_id: str, name: str, backend: Dict, variables: Dict, is_child: bool) -> None:
    where = f"{feature_id}: backend {name!r}"
    if name not in BACKENDS:
        raise ManifestError(f"{where} is not one of {', '.join(BACKENDS)}")
    if name == "inherit" and not is_child:
        raise ManifestError(f"{where}: only subfeatures can inherit")
    unknown = set(backend) - BACKEND_KEYS
    if unknown:
        raise ManifestError(f"{where} has unknown keys {sorted(unknown)}")
    if backend.get("threads", "global") not in ("global", "feature"):
        raise ManifestError(f"{where}: threads must be 'global' or 'feature'")
    for alternative in backend.get("build", []):
        for variable, value in alternative.items():
            if variable not in variables:
                raise ManifestError(f"{where} uses unknown build variable {variable!r}")
            if value not in variables[variable]["values"]:
                raise ManifestError(f"{where}: {variable} has no value {value!r}")


def load(manifest_paths: List[Path], build_variables: Dict) -> Catalog:
    catalog = Catalog()

    def add(entry: Dict, project: str, parent: str) -> str:
        fid = entry.get("id")
        if not fid or fid in catalog.features:
            raise ManifestError(f"Feature id {fid!r} is missing or repeated")
        backends = entry.get("backends") or {}
        if not backends:
            raise ManifestError(f"{fid} declares no backends")
        for name, backend in backends.items():
            _check_backend(fid, name, backend, build_variables, bool(parent))
        default = entry.get("default", next(iter(backends)))
        if default not in backends:
            raise ManifestError(f"{fid}: default backend {default!r} is not offered")
        feature = Feature(fid, entry.get("title", fid), project, entry.get("description", ""),
                          default, backends, parent)
        catalog.features[fid] = feature
        for child in entry.get("subfeatures", []):
            feature.children.append(add(child, project, fid))
        return fid

    for path in manifest_paths:
        data = json.loads(Path(path).read_text())
        project = data["project"]
        top = [add(entry, project, "") for entry in data.get("features", [])]
        catalog.projects.append({"id": project, "title": data.get("title", project), "features": top})
        for rule in data.get("rules", []):
            for fid in rule.get("when", {}):
                if fid not in catalog.features:
                    raise ManifestError(f"{path}: rule refers to unknown feature {fid!r}")
            catalog.rules.append(rule)
    for feature in catalog.features.values():
        for backend in feature.backends.values():
            for fid in backend.get("requires", {}):
                if fid not in catalog.features:
                    raise ManifestError(f"{feature.id} requires unknown feature {fid!r}")
    return catalog


@dataclass
class Resolved:
    """One feature's effective execution."""
    choice: str        # what was selected, possibly "inherit"
    backend: str       # serial, openmp or gpu
    threads: int       # host threads (openmp), 1 for serial, 0 for gpu
    gpus: int          # GPUs (gpu), else 0
    threads_scope: str = "global"


@dataclass
class Resolution:
    features: Dict[str, Resolved] = field(default_factory=dict)
    build: Dict[str, str] = field(default_factory=dict)    # build variable -> value
    build_flags: List[str] = field(default_factory=list)
    pool_threads: int = 1                                  # OMP_NUM_THREADS
    max_feature_gpus: int = 0
    args: Dict[str, str] = field(default_factory=dict)     # program option -> value
    arg_sources: Dict[str, str] = field(default_factory=dict)  # program option -> feature id
    env: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def uses_gpus(self) -> bool:
        return any(item.backend == "gpu" for item in self.features.values())

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["uses_gpus"] = self.uses_gpus
        return data


def _positive_int(value, default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _solve_build(constraints: List[Tuple[str, List[Dict]]], variables: Dict) -> Optional[Dict[str, str]]:
    """The most preferred build-variable assignment meeting every constraint.

    Preference is the declaration order of variables and of their values.
    A constraint is a list of alternatives; an alternative is met when every
    variable it names has the given value.
    """
    names = list(variables)
    for values in itertools.product(*(list(variables[name]["values"]) for name in names)):
        assignment = dict(zip(names, values))
        if any(assignment.get(variable) not in allowed
               for name in names
               for variable, allowed in variables[name]["values"][assignment[name]].get("requires", {}).items()):
            continue
        if all(any(all(assignment[k] == v for k, v in alternative.items()) for alternative in alternatives)
               for _, alternatives in constraints):
            return assignment
    return None


def resolve(catalog: Catalog, selection: Dict[str, Dict], variables: Dict, *,
            cores_per_task: int, gpus_per_node: int, program_options: Optional[List[str]] = None,
            user_args: str = "") -> Resolution:
    """Resolve the selected backends into effective executions, a build, and run settings.

    program_options are the options the benchmark declares (None: unknown, so
    every feature argument is passed); user_args are the run's own arguments,
    which win over feature arguments.
    """
    out = Resolution()
    selection = selection or {}
    constraints: List[Tuple[str, List[Dict]]] = []

    def named(fid: str) -> str:
        return f"{catalog.features[fid].title} ({fid})"

    def build_text(alternative: Dict[str, str]) -> str:
        return ", ".join(variables[k]["values"][v].get("label", f"{k}={v}") for k, v in alternative.items())

    for fid, feature in catalog.features.items():
        chosen = (selection.get(fid) or {}).get("backend") or feature.default
        if chosen not in feature.backends:
            out.errors.append(f"{named(fid)} cannot run as {chosen!r}; choose "
                              f"{' or '.join(BACKEND_LABEL[b] for b in feature.backends)}.")
            chosen = feature.default
        backend = feature.backends[chosen]
        if backend.get("available", True) is False:
            out.errors.append(f"{named(fid)}: {BACKEND_LABEL[chosen]} is not available yet. "
                              f"{backend.get('note', '')}".strip())
        if chosen == "gpu" and gpus_per_node <= 0:
            out.errors.append(f"{named(fid)} is set to GPU, but the partition has no GPUs.")

        if chosen == "inherit":
            parent = out.features[feature.parent]
            resolved = Resolved("inherit", parent.backend, parent.threads, parent.gpus, parent.threads_scope)
        elif chosen == "openmp":
            threads = _positive_int((selection.get(fid) or {}).get("threads"), cores_per_task)
            if threads < 1 or threads > cores_per_task:
                out.errors.append(f"{named(fid)}: {threads} OpenMP threads is more than the {cores_per_task} "
                                  f"cores per rank. Choose 1 to {cores_per_task}, or give each rank more cores "
                                  f"(step 3).")
                threads = max(1, min(threads, cores_per_task))
            resolved = Resolved("openmp", "openmp", threads, 0, backend.get("threads", "global"))
        elif chosen == "gpu":
            max_gpus = min(int(backend.get("max_gpus", gpus_per_node)), gpus_per_node) if gpus_per_node else 0
            gpus = _positive_int((selection.get(fid) or {}).get("gpus"), 1)
            if gpus_per_node and not 1 <= gpus <= max_gpus:
                out.errors.append(f"{named(fid)}: {gpus} GPUs per rank is more than it can use; choose 1 to {max_gpus}.")
            resolved = Resolved("gpu", "gpu", 0, gpus, "global")
        else:
            resolved = Resolved("serial", "serial", 1, 0, backend.get("threads", "global"))
        out.features[fid] = resolved

        if backend.get("build"):
            constraints.append((f"{named(fid)} on {BACKEND_LABEL[chosen]}", backend["build"]))
        values = {"threads": resolved.threads, "gpus": resolved.gpus}
        for option, template in backend.get("args", {}).items():
            out.args[option] = str(template).format(**values)
            out.arg_sources[option] = fid
        for name, template in backend.get("env", {}).items():
            out.env[name] = str(template).format(**values)

    for fid, feature in catalog.features.items():
        backend = feature.backends.get(out.features[fid].choice, {})
        for other, allowed in backend.get("requires", {}).items():
            if out.features[other].backend not in allowed:
                out.errors.append(f"{named(fid)} on {BACKEND_LABEL[out.features[fid].choice]} needs "
                                  f"{named(other)} on {' or '.join(BACKEND_LABEL[a] for a in allowed)}, not "
                                  f"{BACKEND_LABEL[out.features[other].backend]}.")

    build = _solve_build(constraints, variables)
    if build is None:
        needs = "; ".join(f"{label} needs a build " + " or ".join(build_text(a) for a in alternatives)
                          for label, alternatives in constraints)
        out.errors.append(f"No single build of RICH fits these choices: {needs}. Change one of them.")
    else:
        out.build = build
        for name, value in build.items():
            out.build_flags += variables[name]["values"][value].get("flags", [])

    openmp = {fid: r for fid, r in out.features.items() if r.backend == "openmp"}
    out.pool_threads = max([r.threads for r in openmp.values()] + [1])
    out.max_feature_gpus = max([r.gpus for r in out.features.values()] + [0])
    kokkos_build = bool(out.build) and out.build.get("kokkos", "none") != "none"
    if out.pool_threads > 1:
        for fid, r in out.features.items():
            # A feature with a single backend only describes what the code does; nothing to warn about
            if (r.threads_scope != "global" or r.choice == "inherit" or r.backend == "gpu"
                    or len(catalog.features[fid].backends) == 1):
                continue
            if r.backend == "openmp" and r.threads < out.pool_threads:
                out.warnings.append(f"{named(fid)} asks for {r.threads} threads, but it uses all "
                                    f"{out.pool_threads} threads of the rank (OMP_NUM_THREADS={out.pool_threads}): "
                                    f"its code cannot use its own thread count yet.")
            elif r.backend == "serial" and kokkos_build:
                out.warnings.append(f"{named(fid)} is set to Serial, but other parts use OpenMP, so it runs on "
                                    f"the rank's {out.pool_threads} threads too (OMP_NUM_THREADS="
                                    f"{out.pool_threads}). Set it to OpenMP to say so.")

    for rule in catalog.rules:
        if all(out.features[fid].backend in allowed for fid, allowed in rule["when"].items()):
            (out.notes if rule.get("level") == "note" else out.warnings).append(rule["message"])

    if program_options is not None:
        declared = set(program_options)
        for option in list(out.args):
            if option not in declared:
                source = out.arg_sources.pop(option)
                out.args.pop(option)
                if out.features[source].backend != "serial":
                    out.warnings.append(f"{named(source)}: this benchmark has no --{option} option, so the "
                                        f"setting reaches the code only through the execution config.")
    for option in list(out.args):
        if re.search(rf"(^|\s)--{re.escape(option)}(=|\s|$)", user_args or ""):
            source = out.arg_sources.pop(option)
            out.args.pop(option)
            if out.features[source].backend != "serial":
                out.warnings.append(f"Your program arguments set --{option} themselves; "
                                    f"{named(source)} does not override it.")
    return out


def execution_config(catalog: Catalog, resolution: Resolution, header: str = "") -> str:
    """The file RICH reads through utils/ExecutionConfig.hpp (RICH_EXECUTION_CONFIG)."""
    lines = ["# RICH execution config: feature backend threads gpus"]
    if header:
        lines.append(f"# {header}")
    for fid in catalog.features:
        r = resolution.features[fid]
        lines.append(f"{fid} {r.backend} {r.threads} {r.gpus}")
    return "\n".join(lines) + "\n"


_USE = re.compile(r'rich::execution::(?:Get|OnDevice|Threads|BackendOf)\s*\(\s*"([^"]+)"')


def scan_sources(roots: List[Path], catalog: Catalog) -> Dict[str, List[str]]:
    """Feature ids the C++ code asks the execution config for, compared with the manifests.

    Returns {"used": [...], "unknown": ids the code uses but no manifest declares,
    "unread": declared ids no code reads yet}.
    """
    used: Dict[str, List[str]] = {}
    for root in roots:
        for path in root.rglob("*"):
            # ExecutionConfig.hpp only shows usage examples
            if (path.suffix not in {".cpp", ".hpp", ".h", ".cc", ".cxx"} or path.name == "ExecutionConfig.hpp"
                    or not path.is_file()):
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            if "rich::execution::" not in text:
                continue
            for fid in _USE.findall(text):
                used.setdefault(fid, []).append(str(path))
    return {"used": sorted(used), "unknown": sorted(set(used) - set(catalog.features)),
            "unread": sorted(set(catalog.features) - set(used)), "locations": used}
