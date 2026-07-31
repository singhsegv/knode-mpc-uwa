"""
Workflow:
1. Create MPC instance
2. A PlanarArm() with proper gravity acts as the "dynamics model" in the MPC class for prediction
3. Another PlanarArm() with gravity set to 2x acts as the "Plant", so there's a mismatch
4. KNODE learns the residual on that Plant gravity settings but with GT data collected from 1x gravity
5. Boom, we're done
"""
from robots.simulator import PlanarArm


if __name__ == "__main__":
    plant = PlanarArm(gravity=[0.0, -9.81, 0.0])
    mpc_dynamics = PlanarArm(gravity=[0.0, -18.81, 0.0])
