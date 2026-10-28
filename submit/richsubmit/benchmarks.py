"""Benchmarks: the problem directories under runs/ that build_rich.sh can compile.

What a benchmark accepts is read from its sources, without building it:
  - command-line options it declares through utils/arguments/ArgumentParser
      arguments.addOption<int>("n", o.n, "cells per Cartesian axis");  arguments.addFlag("octant", "...");
  - environment variables it declares through utils/arguments/EnvironmentParser
      environment.addRequired<std::string>("RICH_OUTPUT_DIR", "directory for the results");
      environment.addOptional<int>("STORM_SEED", 7, "Monte Carlo seed");
The sources are the benchmark's own files plus the headers under runs/ they include
(e.g. runs/mc_results_dir.hpp), followed recursively. Comments are skipped, and names and
descriptions must be string literals.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

SOURCE_SUFFIXES = {".cpp", ".cxx", ".cc"}

# One string literal, or adjacent ones that C++ joins: "Courant " "factor"
_STRING = r'((?:"(?:[^"\\]|\\.)*"\s*)+)'
_OPTION = re.compile(r'addOption\s*<\s*([^>]+?)\s*>\s*\(\s*"([^"]+)"\s*,\s*(.*?)\s*,\s*' + _STRING, re.S)
_FLAG = re.compile(r'addFlag\s*\(\s*"([^"]+)"\s*,\s*' + _STRING, re.S)
_ENV_REQUIRED = re.compile(r'addRequired\s*<\s*([^>]+?)\s*>\s*\(\s*"([^"]+)"\s*(?:,\s*' + _STRING + r')?\s*\)', re.S)
_ENV_OPTIONAL = re.compile(r'addOptional\s*<\s*([^>]+?)\s*>\s*\(\s*"([^"]+)"\s*,\s*(.*?)\s*(?:,\s*' + _STRING +
                           r')?\s*\)\s*;', re.S)
_INCLUDE = re.compile(r'#\s*include\s*"([^"]+)"')


@dataclass
class ProgramOption:
    name: str
    type: str        # C++ type, or "flag"
    help: str


@dataclass
class EnvVariable:
    name: str
    type: str        # C++ type
    required: bool
    default: str     # the C++ expression of the default; empty when required
    help: str


@dataclass
class Benchmark:
    name: str
    path: str
    description: str = ""
    options: List[ProgramOption] = field(default_factory=list)
    environment: List[EnvVariable] = field(default_factory=list)

    @property
    def required_env(self) -> List[str]:
        return [variable.name for variable in self.environment if variable.required]

    def to_dict(self) -> Dict:
        data = asdict(self)
        data["required_env"] = self.required_env
        return data


def _description(directory: Path) -> str:
    """The first prose line of the directory's README, if any."""
    for readme in ("README.md", "README", "readme.md"):
        path = directory / readme
        if not path.is_file():
            continue
        for line in path.read_text(errors="replace").splitlines():
            text = line.strip().lstrip("#").strip()
            if text:
                return text
    return ""


def _strip_comments(text: str) -> str:
    """C++ text without // and /* */ comments; string and character literals are kept."""
    return re.sub(r'//[^\n]*|/\*.*?\*/|("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')',
                  lambda match: match.group(1) or " ", text, flags=re.S)


def _sources(directory: Path, runs_dir: Path) -> List[str]:
    """The benchmark's sources and the runs/ headers they include, without comments."""
    runs_dir = runs_dir.resolve()
    pending = [path for path in sorted(directory.iterdir()) if path.suffix in SOURCE_SUFFIXES]
    seen, texts = set(), []
    while pending:
        path = pending.pop(0).resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        text = _strip_comments(path.read_text(errors="replace"))
        texts.append(text)
        for include in _INCLUDE.findall(text):
            for base in (path.parent, runs_dir.parent):  # "local.hpp" and "runs/shared.hpp"
                candidate = (base / include).resolve()
                if runs_dir in candidate.parents and candidate.is_file():
                    pending.append(candidate)
                    break
    return texts


def _clean(text: str) -> str:
    """The text of (adjacent) string literals, or of an expression, on one line."""
    if text.lstrip().startswith('"'):
        text = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', text))
    return " ".join(text.split())


def _program_options(texts: List[str]) -> List[ProgramOption]:
    options: List[ProgramOption] = []
    for text in texts:
        options += [ProgramOption(name, _clean(kind), _clean(help_text))
                    for kind, name, _, help_text in _OPTION.findall(text)]
        options += [ProgramOption(name, "flag", _clean(help_text)) for name, help_text in _FLAG.findall(text)]
    return options


def _environment(texts: List[str]) -> List[EnvVariable]:
    variables: Dict[str, EnvVariable] = {}
    for text in texts:
        for kind, name, help_text in _ENV_REQUIRED.findall(text):
            variables[name] = EnvVariable(name, _clean(kind), True, "", _clean(help_text))
        for kind, name, default, help_text in _ENV_OPTIONAL.findall(text):
            if name not in variables:  # a required declaration wins
                variables[name] = EnvVariable(name, _clean(kind), False, _clean(default), _clean(help_text))
    return sorted(variables.values(), key=lambda variable: (not variable.required, variable.name))


def discover(runs_dir: Path, with_options: bool = False) -> List[Benchmark]:
    """Directories of runs_dir that hold C++ sources, sorted case-insensitively."""
    found = []
    for directory in runs_dir.iterdir():
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        if not any(entry.suffix in SOURCE_SUFFIXES for entry in directory.iterdir()):
            continue
        benchmark = Benchmark(directory.name, str(directory), _description(directory))
        if with_options:
            texts = _sources(directory, runs_dir)
            benchmark.options, benchmark.environment = _program_options(texts), _environment(texts)
        found.append(benchmark)
    return sorted(found, key=lambda benchmark: benchmark.name.lower())


def find(runs_dir: Path, name: str) -> Optional[Benchmark]:
    directory = runs_dir / name
    if not name or "/" in name or not directory.is_dir():
        return None
    texts = _sources(directory, runs_dir)
    return Benchmark(name, str(directory), _description(directory), _program_options(texts), _environment(texts))
