# import libraries
import torch
import matplotlib.pyplot as plt
import argparse
import os
import yaml

plt.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1F1F1F'
plt.rcParams['axes.facecolor'] = '#1F1F1F'
plt.rcParams['savefig.facecolor'] = '#1F1F1F'
torch.set_float32_matmul_precision('medium')

# read in all config files


class DotDict(dict):
    def __init__(self, mapping=None):
        super().__init__()
        mapping = mapping or {} 
        for key, value in mapping.items():
            self[key] = DotDict(value) if isinstance(value, dict) else value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Key '{key}' not in config")

    def __setattr__(self, key, value):
        self[key] = value

def load_yaml_as_dotdict(filepath):
    with open(filepath, "r") as file:
        data = yaml.safe_load(file) or {}  # for if yaml empty lined
    return DotDict(data)

parser = argparse.ArgumentParser()
parser.add_argument("--CB", type=str, default="std")
parser.add_argument("--CD", type=str, default="std")
parser.add_argument("--CM", type=str, default="std")
parser.add_argument("--CT", type=str, default="std")
parser.add_argument("--trainer", type=str, default="MTT")
parser.add_argument("--out", type=str, default=None)
args = parser.parse_args()

if os.path.exists("conf/base/" + args.CB + ".yaml"):
    cb = load_yaml_as_dotdict("conf/base/" + args.CB + ".yaml")
else:
    raise FileNotFoundError(f"Config file {args.CB}.yaml not found.")
if os.path.exists("conf/data/" + args.CD + ".yaml"):
    cd = load_yaml_as_dotdict("conf/data/" + args.CD + ".yaml")
else:
    raise FileNotFoundError(f"Config file {args.CD}.yaml not found.")
if os.path.exists("conf/model/" + args.CM + ".yaml"):
    cm = load_yaml_as_dotdict("conf/model/" + args.CM + ".yaml")
else:
    raise FileNotFoundError(f"Config file {args.CM}.yaml not found.")
if os.path.exists("conf/training/" + args.CT + ".yaml"):
    ct = load_yaml_as_dotdict("conf/training/" + args.CT + ".yaml")
else:
    raise FileNotFoundError(f"Config file {args.CT}.yaml not found.")
print("Configs loaded.")
raise NotImplementedError("Testing script not yet implemented.")
# import data

# load model


def calculate_ss_error_per_dataset():
    # absolute and relative error per dataset?
    pass

def calculate_rollout_error_per_dataset():
    # not sure yet
    pass

def calculate_spectra_plots_per_dataset():
    # not sure yet
    pass

if __name__ == "__main__":
    calculate_ss_error_per_dataset()