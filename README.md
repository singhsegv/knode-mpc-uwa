# Learning 2D Planar Arm Dynamics 

## Running Experiments
```
python run_experiment.py --dataset evaluation1 --model basic1

# optional: --epochs 500 --lr 0.001

python run_experiment.py --dataset evaluation1 --model basic1 --name testing

# Dataset Generation
python utils/training_data_generator.py --viz --custom-dynamics
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
