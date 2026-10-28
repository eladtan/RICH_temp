"""Unit tests without SLURM: python3 -m unittest discover -s submit/tests"""

import subprocess
import sys
import unittest
from pathlib import Path

SUBMIT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUBMIT))

from richsubmit import Context, Resources, RunSpec, make_plan  # noqa: E402
from richsubmit import layout, slurm  # noqa: E402

GPU_NODE = slurm.Partition("gpu", "UP", True, 100, "2-00:00:00", "01:00:00", [], 112, 2, 56, 512000, 8)
CPU_NODE = slurm.Partition("cpu", "UP", False, 100, "1-00:00:00", "01:00:00", [], 128, 1, 128, 256000, 0)
DEBUG = slurm.Partition("debug", "UP", False, 4, "02:00:00", "00:30:00", ["develop"], 112, 2, 56, 512000, 8)


def context():
    ctx = Context()
    ctx._partitions = [GPU_NODE, CPU_NODE, DEBUG]
    ctx._cluster = "frontier"
    return ctx


def spec(**changes):
    data = {"partition": "gpu", "time_limit": "01:00:00", "account": "acct", "benchmark": "RMTV",
            "resources": {"ntasks": 8, "cores_per_task": 7}, "features": {}}
    for key, value in changes.items():
        if key in ("resources", "features"):
            data[key] = dict(data[key], **value)
        else:
            data[key] = value
    return RunSpec.from_dict(data)


class Parsing(unittest.TestCase):
    def test_time(self):
        self.assertEqual(slurm.parse_time_minutes("7-00:00:00"), 7 * 24 * 60)
        self.assertEqual(slurm.parse_time_minutes("02:30:00"), 150)
        self.assertEqual(slurm.parse_time_minutes("45"), 45)
        self.assertIsNone(slurm.parse_time_minutes("UNLIMITED"))
        self.assertIsNone(slurm.parse_time_minutes("soon"))

    def test_memory_and_gres(self):
        self.assertEqual(slurm.parse_memory_mb("500G"), 512000)
        self.assertEqual(slurm.parse_memory_mb("512000"), 512000)
        self.assertEqual(slurm.parse_gres_gpus("gpu:8(S:0-7)"), (8, ""))
        self.assertEqual(slurm.parse_gres_gpus("gpu:a100:4(S:0),shard:8"), (4, "a100"))
        self.assertEqual(slurm.parse_gres_gpus("(null)"), (0, ""))


class Layouts(unittest.TestCase):
    def test_cores_bound_and_even_spread(self):
        placed = layout.compute(Resources(ntasks=100, cores_per_task=1), GPU_NODE, False)
        self.assertEqual((placed.nodes, placed.ranks_per_node), (2, 50))

    def test_min_memory_per_rank(self):
        placed = layout.compute(Resources(ntasks=20, cores_per_task=1, min_memory_per_rank_mb=100000),
                                GPU_NODE, False)
        self.assertEqual(placed.max_ranks_per_node, 5)
        self.assertIn("memory", placed.limited_by)
        self.assertEqual(placed.nodes, 4)
        self.assertEqual(placed.memory_per_node_mb, 500000)

    def test_ranks_per_node_too_many(self):
        placed = layout.compute(Resources(ntasks=64, cores_per_task=7, ranks_per_node=9), GPU_NODE, False)
        self.assertTrue(any("exceed" in error for error in placed.errors))

    def test_gpus_per_rank_capped_by_node(self):
        placed = layout.compute(Resources(ntasks=2, cores_per_task=1, gpus_per_rank=9), GPU_NODE, True)
        self.assertTrue(any("another node" in error for error in placed.errors))

    def test_oversubscription(self):
        placed = layout.compute(Resources(ntasks=32, cores_per_task=3, gpus_per_rank=1, ranks_per_gpu=2),
                                GPU_NODE, True)
        self.assertEqual((placed.nodes, placed.ranks_per_node, placed.gpus_per_node), (2, 16, 8))
        placed = layout.compute(Resources(ntasks=4, cores_per_task=1, gpus_per_rank=2, ranks_per_gpu=2),
                                GPU_NODE, True)
        self.assertTrue(placed.errors)


class Plans(unittest.TestCase):
    def setUp(self):
        self.ctx = context()

    def test_gpu_rmtv(self):
        plan = make_plan(self.ctx, spec(features={"storm.transport": {"backend": "gpu"},
                                                  "madvoro.build": {"backend": "openmp", "threads": 7}}))
        self.assertTrue(plan.ok, plan.errors)
        self.assertEqual(plan.resolution["build"], {"kokkos": "device", "geometry_on_host": "on"})
        self.assertIn("--with-gpu", plan.build_command)
        self.assertIn("bind_gpus.sh", plan.launch_command)
        self.assertEqual(plan.environment["OMP_NUM_THREADS"], "7")
        self.assertIn("storm.transport gpu 0 1", plan.execution_config)

    def test_no_gpus_on_cpu_partition(self):
        plan = make_plan(self.ctx, spec(partition="cpu", features={"storm.transport": {"backend": "gpu"}}))
        self.assertTrue(any("no GPUs" in error for error in plan.errors))

    def test_threads_capped_by_cores_per_task(self):
        plan = make_plan(self.ctx, spec(features={"madvoro.build": {"backend": "openmp", "threads": 8}}))
        self.assertTrue(any("8 OpenMP threads" in error for error in plan.errors))

    def test_feature_gpus_capped(self):
        plan = make_plan(self.ctx, spec(features={"storm.transport": {"backend": "gpu", "gpus": 2}}))
        self.assertTrue(any("choose 1 to 1" in error for error in plan.errors))  # Kokkos uses one device

    def test_build_conflict(self):
        plan = make_plan(self.ctx, spec(features={"storm.transport": {"backend": "openmp"},
                                                  "madvoro.build.geometry": {"backend": "gpu"}}))
        self.assertTrue(any("No single build" in error for error in plan.errors))

    def test_subdomains_argument(self):
        plan = make_plan(self.ctx, spec(features={"madvoro.build.subdomains": {"backend": "openmp", "threads": 4}}))
        self.assertTrue(plan.ok, plan.errors)
        self.assertIn("--voronoi-subdomains 4", plan.launch_command)
        self.assertEqual(plan.resolution["build"]["kokkos"], "host")

    def test_user_argument_wins(self):
        plan = make_plan(self.ctx, spec(program_args="--voronoi-subdomains 2",
                                        features={"madvoro.build.subdomains": {"backend": "openmp", "threads": 4}}))
        self.assertNotIn("--voronoi-subdomains 4", plan.launch_command)
        self.assertTrue(any("does not override" in warning for warning in plan.warnings))

    def test_unavailable_backend(self):
        plan = make_plan(self.ctx, spec(features={"rich.hydro.slopes": {"backend": "gpu"}}))
        self.assertTrue(any("not available yet" in error for error in plan.errors))

    def test_serial_run(self):
        plan = make_plan(self.ctx, spec(build_config="gnuRelease", resources={"ntasks": 1, "cores_per_task": 1}))
        self.assertTrue(plan.ok, plan.errors)
        self.assertFalse(plan.launch_command.startswith("mpirun"))
        plan = make_plan(self.ctx, spec(build_config="gnuRelease", resources={"ntasks": 4}))
        self.assertTrue(any("MPI build" in error for error in plan.errors))

    def test_site_node_limit(self):
        service = slurm.Partition("service", "UP", False, 64, "7-00:00:00", "01:00:00", ["develop"], 112, 2, 56,
                                  512000, 8)
        self.ctx._partitions.append(service)
        plan = make_plan(self.ctx, spec(partition="service", qos="develop", resources={"ntasks": 16, "cores_per_task": 7}))
        self.assertTrue(any("at most 1 node" in error for error in plan.errors))
        plan = make_plan(self.ctx, spec(partition="service", qos="develop"))
        self.assertTrue(plan.ok, plan.errors)
        self.assertNotIn("RICH_OUTPUT_DIR", plan.run_script)  # nothing chosen for the user

    def test_required_environment(self):
        plan = make_plan(self.ctx, spec(benchmark="Elad_paper_Marshak"))
        self.assertTrue(any("needs RICH_OUTPUT_DIR" in i["text"] and i["step"] == "problem" for i in plan.issues))
        plan = make_plan(self.ctx, spec(benchmark="Elad_paper_Marshak",
                                        environment={"RICH_OUTPUT_DIR": "/scratch/$USER/out/$SLURM_JOB_ID"}))
        self.assertTrue(plan.ok, plan.errors)
        self.assertIn('export RICH_OUTPUT_DIR="/scratch/$USER/out/$SLURM_JOB_ID"', plan.run_script)

    def test_environment_names(self):
        plan = make_plan(self.ctx, spec(environment={"1BAD": "x", "OMP_NUM_THREADS": "3",
                                                     "ROCR_VISIBLE_DEVICES": "0", "": "ignored", "FINE": "y"}))
        texts = " ".join(plan.errors)
        self.assertIn("'1BAD' is not a valid", texts)
        self.assertIn("OMP_NUM_THREADS is set by the tool", texts)
        self.assertIn("ROCR_VISIBLE_DEVICES is set by the tool", texts)
        self.assertEqual(plan.user_environment.get("FINE"), "y")

    def test_environment_quoting(self):
        value = 'a "quoted" `tick` \\ back $HOME'
        plan = make_plan(self.ctx, spec(environment={"MY_VALUE": value}))
        line = next(l for l in plan.run_script.splitlines() if l.startswith("export MY_VALUE="))
        out = subprocess.run(["bash", "-c", line + '; printf %s "$MY_VALUE"'], stdout=subprocess.PIPE,
                             universal_newlines=True, env={"HOME": "/home/x", "PATH": "/usr/bin:/bin"}).stdout
        self.assertEqual(out, value.replace("$HOME", "/home/x"))

    def test_partition_limits(self):
        plan = make_plan(self.ctx, spec(partition="debug", time_limit="03:00:00"))
        self.assertTrue(any("exceeds" in error for error in plan.errors))
        self.assertTrue(any("QOS" in error for error in plan.errors))
        plan = make_plan(self.ctx, spec(partition="debug", qos="develop"))
        self.assertTrue(plan.ok, plan.errors)


class Declarations(unittest.TestCase):
    def test_scan(self):
        import tempfile
        from richsubmit import benchmarks
        with tempfile.TemporaryDirectory() as root:
            runs = Path(root) / "runs"
            (runs / "demo").mkdir(parents=True)
            (runs / "shared.hpp").write_text(
                'inline void f() { EnvironmentParser e; e.addRequired<std::string>("OUT_DIR", "results"); }\n')
            (runs / "demo" / "main.cpp").write_text("""#include "runs/shared.hpp"
// environment.addRequired<int>("IN_A_COMMENT", "not real");
/* arguments.addOption<int>("hidden", 1, "not real"); */
int main(int argc, char **argv) {
    ArgumentParser arguments("demo");
    arguments.addOption<double>("cfl", o.cfl, "Courant "
                                "factor");
    arguments.addFlag("quiet", "say less");
    EnvironmentParser environment("demo");
    environment.addOptional<int>("SEED", 7, "Monte Carlo seed");
    environment.addOptional<std::string>("OUT_DIR", "x", "declared twice; required wins");
    environment.addOptional<bool>("NO_HELP", false);
}
""")
            found = benchmarks.find(runs, "demo")
            self.assertEqual([(o.name, o.type) for o in found.options], [("cfl", "double"), ("quiet", "flag")])
            env = {v.name: v for v in found.environment}
            self.assertEqual(sorted(env), ["NO_HELP", "OUT_DIR", "SEED"])
            self.assertTrue(env["OUT_DIR"].required)
            self.assertEqual((env["SEED"].default, env["SEED"].help), ("7", "Monte Carlo seed"))
            self.assertEqual((env["NO_HELP"].default, env["NO_HELP"].help), ("false", ""))
            self.assertEqual(found.required_env, ["OUT_DIR"])


class Issues(unittest.TestCase):
    def test_messages_carry_their_step(self):
        plan = make_plan(context(), spec(benchmark="", time_limit="soon",
                                         features={"madvoro.build": {"backend": "openmp", "threads": 8}}))
        steps = {issue["step"] for issue in plan.issues if issue["level"] == "error"}
        self.assertEqual(steps, {"partition", "problem", "features"})
        self.assertEqual(len(plan.issues), len(plan.errors) + len(plan.warnings) + len(plan.notes))

    def test_checks_are_not_listed_as_jobs(self):
        import tempfile
        from richsubmit import jobs
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / ".check_20260101_000000_RMTV").mkdir()
            (Path(directory) / "20260101_000000_RMTV").mkdir()
            self.assertEqual([job["name"] for job in jobs.recent(Path(directory))], ["20260101_000000_RMTV"])


class GpuBinding(unittest.TestCase):
    def bind(self, local_rank, **env):
        script = SUBMIT / "bind_gpus.sh"
        environment = {"PATH": "/usr/bin:/bin", "OMPI_COMM_WORLD_LOCAL_RANK": str(local_rank),
                       "RICH_SUBMIT_GPU_ENV": "ROCR_VISIBLE_DEVICES"}
        environment.update({key: str(value) for key, value in env.items()})
        return subprocess.run([str(script), "printenv", "ROCR_VISIBLE_DEVICES"], env=environment,
                              stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.strip()

    def test_frontier_order(self):
        order = "4 5 2 3 6 7 0 1"
        self.assertEqual(self.bind(0, RICH_SUBMIT_GPU_ORDER=order), "4")
        self.assertEqual(self.bind(6, RICH_SUBMIT_GPU_ORDER=order), "0")
        self.assertEqual(self.bind(1, RICH_SUBMIT_GPU_ORDER=order, RICH_SUBMIT_GPUS_PER_RANK=2), "2,3")
        self.assertEqual(self.bind(5, RICH_SUBMIT_GPU_ORDER=order, RICH_SUBMIT_RANKS_PER_GPU=2), "2")

    def test_plain_order(self):
        self.assertEqual(self.bind(3), "3")
        self.assertEqual(self.bind(1, RICH_SUBMIT_GPUS_PER_RANK=4), "4,5,6,7")


if __name__ == "__main__":
    unittest.main()
