import math
import random
import numpy as np


NUM_TARGET = 4
NUM_SCAN = 150
TARGET_ENTER = [x - 1 for x in [1, 21, 41, 61]]
TARGET_EXIT = [x - 1 for x in [76, 96, 116, 136]]

EXPECTED_FALSE_TARGETS = 30

SIGMA_P = 0.15
DELTA_T = 1.0
DELTA_T_SQUARE = DELTA_T * DELTA_T
DELTA_T_CUBE = DELTA_T * DELTA_T_SQUARE

F = np.array([
        [1., 0., DELTA_T, 0.],
        [0., 1., 0., DELTA_T],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])

Q = SIGMA_P * SIGMA_P * np.array([
        [DELTA_T_CUBE / 3., 0., DELTA_T_SQUARE / 2., 0.],
        [0., DELTA_T_CUBE / 3., 0., DELTA_T_SQUARE / 2.],
        [DELTA_T_SQUARE / 2., 0., DELTA_T, 0.],
        [0., DELTA_T_SQUARE / 2., 0., DELTA_T]])

P_D_S = 0.95   # p^D(s)
P_D_PHI = 0.99  # p^D(phi)
OBS_COV = np.array([[100., 0.], [0., 100.]])


class Simulator:
    """
    A simulator generating the ground truth tracks and the measurements of targets.
    """
    def __init__(self, random_seed=None):
        self.seed = random_seed
        random.seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

    def init_state(self) -> np.ndarray:
        """
        Sample a initial state (x1, x2, v1, v2) where
        x1 ~ Uniform(Union([-800, -400], [400, 800])) 
        x2 ~ Uniform(Union([-800, -400], [400, 800]))
        v1 ~ Uniform([-10, 0]) if x1 > 0 else Uniform([0, 10])
        v2 ~ Uniform([-10, 0]) if x2 > 0 else Uniform([0, 10])
        """
        x1 = self.sample_x()
        x2 = self.sample_x()
        v1 = self.sample_v(x1)
        v2 = self.sample_v(x2)
        return np.array([x1, x2, v1, v2])

    def sample_x(self) -> float:
        """
        Approximately implement
        x ~ Uniform(Union([-800, -400], [400, 800]))
        """
        x = self.rng.uniform(0, 800)
        return x if x > 400 else x - 800

    def sample_v(self, x: float) -> float:
        """
        Approximately implement
        v ~ Uniform([-10, 0]) if x > 0 else Uniform([0, 10])
        """
        return self.rng.uniform(-10, 0) if x > 0 else self.rng.uniform(0, 10)

    def evolve(self, state: np.ndarray) -> np.ndarray:
        """
        Evolve the state according to the transition model:
        an almost constant velocity motion model
        """
        assert state.ndim == 1
        assert state.shape[0] == 4
        mean = np.zeros(state.shape[0])
        cov = Q
        return np.dot(F, state) + self.rng.multivariate_normal(mean, cov)

    def observe(self, true_targets: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        Generate measurements as per the Linear Gaussian observation model
        """
        if true_targets.ndim == 1:
            true_targets = true_targets.reshape(1, -1)
        n_true_targets = true_targets.shape[0] if math.prod(true_targets.shape) > 0 else 0

        observed = []
        cov = OBS_COV

        # observe true targets
        n_obs_true = 0
        for i in range(n_true_targets):
            if self.rng.random() < P_D_S:
                observed.append(self.rng.multivariate_normal(true_targets[i, :2], cov))
                n_obs_true += 1

        # generate and observe false targets
        n_obs_false = 0
        n_false_targets = self.rng.poisson(EXPECTED_FALSE_TARGETS)
        for _ in range(n_false_targets):
            if self.rng.random() < P_D_PHI:
                mean = self.rng.uniform(-800, 800, size=2)
                observed.append(self.rng.multivariate_normal(mean, cov))
                n_obs_false += 1

        # additional info
        info = {
            'TRUE_TARGETS': n_true_targets,
            'TRUE_TARGETS_OBS': n_obs_true,
            'FALSE_TARGETS': n_false_targets,
            'FALSE_TARGETS_OBS': n_obs_false,
        }

        return np.array(observed) if len(observed) > 0 else None, info

    def simulate(self) -> list:
        """
        Simulate the four moving targets as presented in Section 5.4.
        """
        targets = [None for _ in range(NUM_TARGET)]
        sts_all = []
        obs_all = []
        info_all = []

        for n in range(NUM_SCAN):
            # print(f'\n------ Scan {n+1} ------')
            for i in range(NUM_TARGET):
                # move target
                if targets[i] is not None:
                    targets[i][:] = self.evolve(targets[i])

                # target enters
                if n == TARGET_ENTER[i]:
                    targets[i] = self.init_state()
                    # print(f'Target {i+1} enters, location: {targets[i][:2]}')

                # target exits
                if n == TARGET_EXIT[i]:
                    targets[i] = None
                    # print(f'Target {i+1} exits')

            true_targets = [t for t in targets if t is not None]
            # print(f'True targets: {len(true_targets)}')

            if len(true_targets) == 0:
                sts_all.append(None)
            else:
                sts_all.append([t.tolist() for t in true_targets])

            # observe targets
            obs, info = self.observe(np.array(true_targets))
            obs_all.append(obs)
            info_all.append(info)

            # print(f'Observed targets: {info["TRUE_TARGETS_OBS"]}')

        # print(f'\nTotal number of observations: {len(obs_all)}\n')

        return sts_all, obs_all, info_all

