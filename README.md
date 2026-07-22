# Learning 2D Planar Arm Dynamics 

## Running Experiments
```
python run_experiment.py --dataset evaluation1 --model basic1

# optional: --epochs 500 --lr 0.001

python run_experiment.py --dataset evaluation1 --model basic1 --name testing

# Dataset Generation
python utils/training_data_generator.py --viz --custom-dynamics

# Comparing rollouts: EE positions and Tau
python utils/compare_collected_rollouts.py data/standalone data/other_run --save

# Performance and Generalization [Standalone]
python evaluate/benchmark_model.py --run 20260717_171819_evaluation1_basic1_comparison-la-100
python evaluate/benchmark_model.py --run <run_name> --epoch 2100 --dataset benchmark_basic --plot
```


## About
1. Data Collection Harness: `simulate.py`
2. Robot Dynmics Simulator: `dynamics.py`
3. Conda Env Issues: KMP_DUPLICATE_LIB_OK=TRUE python knode-learn-dynamics.py
    - Torch (from pip) and Pinocchio (from conda) doesn't work together
    - Fix: `conda install -c conda-forge pinocchio pytorch`

## Todo
- Sanity checks for gravity and hold etc like I did for mujoco
- Check urdf: Origins for links and Other stuff for joints
- EE orientation tracking (theta is hardcoded to 0)
- Missing J_dot term in qddot_des
