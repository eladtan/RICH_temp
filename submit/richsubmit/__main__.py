"""Command line: rich-submit [gui|partitions|benchmarks|features|plan|submit] ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import features as features_module
from .plan import Context, RunSpec, make_plan, submit


def _print_json(data) -> None:
    print(json.dumps(data, indent=2))


def _load_spec(path: str) -> RunSpec:
    text = sys.stdin.read() if path == "-" else Path(path).read_text()
    return RunSpec.from_dict(json.loads(text))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rich-submit", description="Configure and submit RICH runs.")
    parser.add_argument("--config", help="Project configuration (default: submit/config.json)")
    parser.add_argument("--port", type=int, default=None,
                        help="Port of the web GUI (default: 8765, or the next free one)")
    commands = parser.add_subparsers(dest="command")

    gui = commands.add_parser("gui", help="Serve the web GUI on this node (the default command)")
    gui.add_argument("--port", dest="gui_port", type=int, default=None,
                     help="Port to listen on (default: 8765, or the next free one)")

    listing = commands.add_parser("partitions", help="SLURM partitions and their node resources")
    listing.add_argument("--json", action="store_true")
    bench = commands.add_parser("benchmarks", help="Benchmarks under runs/")
    bench.add_argument("--json", action="store_true")
    feats = commands.add_parser("features", help="Parallel features from the manifests")
    feats.add_argument("--json", action="store_true")
    feats.add_argument("--check", action="store_true",
                       help="Compare manifest ids with the rich::execution calls in source/ and runs/")

    for name, help_text in (("plan", "Check a spec and print its plan"),
                            ("submit", "Submit a spec")):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("spec", help="Spec JSON file, or - for stdin")
        sub.add_argument("--json", action="store_true")
        if name == "submit":
            sub.add_argument("--dry-run", action="store_true",
                             help="Write the job directory and check it with sbatch --test-only")

    args = parser.parse_args(argv)
    context = Context(args.config)
    command = args.command or "gui"

    if command == "gui":
        from .server import serve
        port = getattr(args, "gui_port", None)
        serve(context, port if port is not None else args.port)
        return 0

    if command == "partitions":
        parts = context.partitions()
        if args.json:
            _print_json([p.to_dict() for p in parts])
        else:
            print(f"{'partition':14} {'state':6} {'nodes':>6} {'cores':>5} {'mem/node':>9} {'GPUs':>4}  max time")
            for p in parts:
                print(f"{p.name + ('*' if p.default else ''):14} {p.state:6} {p.nodes:6} {p.cores_per_node:5} "
                      f"{p.memory_per_node_mb // 1024:7}Gi {p.gpus_per_node:4}  {p.max_time}"
                      + (f"  (QOS {','.join(p.allowed_qos)})" if p.allowed_qos else ""))
        return 0

    if command == "benchmarks":
        found = context.benchmarks(with_options=args.json)
        if args.json:
            _print_json([b.to_dict() for b in found])
        else:
            for b in found:
                print(f"{b.name:34} {b.description}")
        return 0

    if command == "features":
        if args.check:
            report = features_module.scan_sources([context.repo / "source", context.runs_dir], context.catalog)
            _print_json({key: report[key] for key in ("used", "unknown", "unread")})
            return 1 if report["unknown"] else 0
        if args.json:
            _print_json(context.catalog.to_dict())
        else:
            for project in context.catalog.projects:
                print(project["title"])
                stack = [(fid, 1) for fid in reversed(project["features"])]
                while stack:
                    fid, depth = stack.pop()
                    feature = context.catalog.features[fid]
                    offered = ", ".join(name + ("" if backend.get("available", True) else " (n/a)")
                                        for name, backend in feature.backends.items())
                    print(f"{'  ' * depth}{fid:34} [{offered}] default {feature.default}")
                    stack += [(child, depth + 1) for child in reversed(feature.children)]
        return 0

    spec = _load_spec(args.spec)
    if command == "plan":
        plan = make_plan(context, spec)
        if args.json:
            _print_json(plan.to_dict())
        else:
            for error in plan.errors:
                print(f"error: {error}")
            for warning in plan.warnings:
                print(f"warning: {warning}")
            for note in plan.notes:
                print(f"note: {note}")
            if plan.ok:
                print(plan.build_script + plan.run_script if plan.build_script else plan.run_script)
        return 0 if plan.ok else 1

    try:
        result = submit(context, spec, dry_run=args.dry_run)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(result.to_dict())
    else:
        for warning in result.plan["warnings"]:
            print(f"warning: {warning}")
        print(result.message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
