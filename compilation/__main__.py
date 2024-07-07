import argparse
import os
from . import build_program, root_dir

parser = argparse.ArgumentParser(prog="build")

parser.add_argument("--test_name", default="", help="The name of the test to compile")

parser.add_argument("--make_dir", default=root_dir, help="The relative path to the build directory")

parser.add_argument("--energy_groups_num", default="0", help="The number of energy groups for a multigroup run")

# possible configurations
config_params = [
    ["intel", "gnu"],
    ["Release", "Debug"],
    ["", "MPI"],
    ["", "Prof"],
]
from itertools import product
configurations = []
for p in product(*config_params):
    configurations.append("".join(p))
configurations = set(configurations)

def allowed_configuration(string):
    if "all_configurations" == string:
        return configurations
    configs = set(string.split(","))
    assert configs <= configurations, f'{string} is not a combination of elements from {" ".join(configurations)} separated by \",\" or \"all_configurations\"'
    return configs

parser.add_argument("configs",
                    type=allowed_configuration,
                    help="The desired build configurations",
                    metavar=f'config should be any \",\" separated combination of elements from {" ".join(configurations)} separated by \",\" or \"all_configurations\"')

args = parser.parse_args()

assert args.test_name
make_dir = os.path.abspath(args.make_dir)
src_dir = os.path.join(root_dir, 'source')
test_dir = os.path.join(root_dir, f"runs/{args.test_name}")

build_program(configs=args.configs,
                            make_dir=make_dir,
                            src_dir=src_dir,
                            test_dir=test_dir,
                            args=args)

