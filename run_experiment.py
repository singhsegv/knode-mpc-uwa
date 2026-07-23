"""
Orchestrate: load dataset/model configs → generate data if needed → train KNODE.

Usage:
  python run_experiment.py --dataset evaluation1 --model basic1
"""
import argparse
import json
import os
import shutil
import sys
from datetime import datetime

import torch

from knode.learn import train_model
from knode.model import PlanarArmKnodeModel
from utils.data_utils import load_data
from utils.log_utils import Tee
from utils.training_data_generator import collect_from_spec, save_log, plot_collected_trajectory
import knode.learn as learn
import utils.model_utils as model_utils
import utils.visualize_trajectory as visualize_trajectory


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config", "datasets")
MODELS_CONFIG_DIR = os.path.join(PROJECT_ROOT, "config", "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def load_json_config(config_dir, name):
    path = os.path.join(config_dir, f"{name}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return json.load(f), path


def data_dir_has_npz(data_dir):
    if not os.path.isdir(data_dir):
        return False
    return any(name.endswith(".npz") for name in os.listdir(data_dir))


def _generate_split(specs, split_dir):
    os.makedirs(split_dir, exist_ok=True)
    for i, spec in enumerate(specs):
        output_path = os.path.join(split_dir, f"{i:02d}_{spec['type']}.npz")
        log = collect_from_spec(spec, enable_viz=False) 
        save_log(output_path, log)

        plot_collected_trajectory(
            t_log=log["t"], q_log=log["q"],
            trajectory=log["meta"]["trajectory"],
            max_steps=log["meta"]["n_steps"], show=False,
            output_dir=os.path.dirname(output_path)
        )

        print(f"Saved {len(log['t'])} samples -> {output_path}")


def ensure_dataset(dataset_cfg, data_dir):
    """Generate train/test .npz files under data_dir if either split is missing."""
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if data_dir_has_npz(train_dir) and data_dir_has_npz(test_dir):
        print(f"Using existing data in {train_dir} and {test_dir}")
        return

    print(f"Generating train/test data from dataset config into {data_dir}...")
    if not data_dir_has_npz(train_dir):
        _generate_split(dataset_cfg["train_trajectories"], train_dir)
    if not data_dir_has_npz(test_dir):
        _generate_split(dataset_cfg["test_trajectories"], test_dir)


def make_run_dir(dataset_name, model_name, experiment_name=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{stamp}_{dataset_name}_{model_name}"
    if experiment_name:
        run_name = f"{run_name}_{experiment_name}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "outputs"), exist_ok=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run a KNODE training experiment from JSON configs.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset config name under config/datasets/ (without .json)")
    parser.add_argument("--model", type=str, required=True, help="Model config name under config/models/ (without .json)")
    parser.add_argument("--name", type=str, default=None, help="Optional custom experiment name appended to the run directory")
    parser.add_argument("--lookahead", type=int, default=2, help="Override lookahead for NODE (default: 2)")
    parser.add_argument("--epochs", type=int, default=15000, help="Override training epochs (default: 15k)")
    parser.add_argument("--lr", type=float, default=0.001, help="Override learning rate")
    args = parser.parse_args()

    dataset_cfg, dataset_cfg_path = load_json_config(DATASETS_CONFIG_DIR, args.dataset)
    model_cfg, model_cfg_path = load_json_config(MODELS_CONFIG_DIR, args.model)

    data_name = dataset_cfg.get("name", args.dataset)
    data_dir = os.path.join(DATA_DIR, data_name)
    ensure_dataset(dataset_cfg, data_dir)

    run_dir = make_run_dir(args.dataset, args.model, args.name)
    shutil.copy2(dataset_cfg_path, os.path.join(run_dir, "dataset.json"))
    shutil.copy2(model_cfg_path, os.path.join(run_dir, "model.json"))
    print(f"Run directory: {run_dir}")

    model_utils.CHECKPOINT_DIR = os.path.join(run_dir, "checkpoints")
    visualize_trajectory.DEFAULT_OUTPUT_DIR = os.path.join(run_dir, "outputs")
    learn.LOOKAHEAD = args.lookahead
    learn.EPOCHS = args.epochs
    learn.LR = args.lr

    tee = Tee(os.path.join(run_dir, "train.log"))
    sys.stdout = tee

    try:
        print(f"dataset={args.dataset} model={args.model}")
        print(f"lookahead={learn.LOOKAHEAD} epochs={learn.EPOCHS} lr={learn.LR}")
        print(f"model_cfg={model_cfg}")

        train_rollouts = load_data(os.path.join(data_dir, "train"))
        test_rollouts = load_data(os.path.join(data_dir, "test"))

        train_data = train_rollouts
        test_data = test_rollouts[0] # TODO: FIX ME LATER

        # TODO: DATA AND NODE STEP_SIZE SHOULD MATCH
        model = PlanarArmKnodeModel(model_cfg).to(torch.float64)
        train_model(model, train_data, test_data)
        
        print(f"Finished. Artifacts in {run_dir}")
    finally:
        sys.stdout = tee._stdout
        tee.close()


if __name__ == "__main__":
    main()
