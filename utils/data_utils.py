import os
import numpy as np


def load_data(data_path):
    """
    Each file is expected to contain:
      - arrays: t, q, qdot, tau, dt
      - meta: dict with trajectory parameters
        
    data = np.hstack([t, q, qdot, tau])

    Returns "list" of concatenated [t, q, qdot, tau] rows across all trajectories -> 1 for each 20s rollout
    """
    def _load_npz_trajectory(path):
        with np.load(path, allow_pickle=True) as npz:
            t = npz["t"][:, np.newaxis]
            q = npz["q"]
            qdot = npz["qdot"]
            tau = npz["tau"]
            # meta = npz["meta"].item()
            trajectory = np.hstack([t, q, qdot, tau])

        return trajectory

    def _list_npz_files(data_path):
        if os.path.isdir(data_path):
            npz_paths = sorted(
                os.path.join(data_path, name)
                for name in os.listdir(data_path)
                if name.endswith(".npz")
            )
            if not npz_paths:
                raise FileNotFoundError(f"No .npz files found in {data_path}")
            return npz_paths
        return [data_path]

    trajectories = []
    for path in _list_npz_files(data_path):
        trajectory = _load_npz_trajectory(path)
        trajectories.append(trajectory)

    # TODO: This doesn't make sense for now -> Use each rollout separately
    # combined_trajectories = np.vstack(trajectories)
    # return combined_trajectories
    return trajectories