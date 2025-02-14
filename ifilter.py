import math
import numpy as np
from tqdm import tqdm
from simulator import (
    F,
    Q,
    P_D_S,
    P_D_PHI,
    OBS_COV,
)

# swap the values of q(s | phi) and q(phi | s) in Eq. (5.37),
# so that Eq. (5.11) is satisfied when x = phi.
# Q_S_PHI = 2e-4 / 180000.  # q(s | phi)
# Q_PHI_S = 1e-6            # q(phi | s)
Q_S_PHI = 1e-6            # q(s | phi)
Q_PHI_S = 2e-4 / 180000.  # q(phi | s)
Q_PHI_PHI = 1. - 2e-4     # q(phi | phi)
L_Y_PHI = 1. / (1600**2)  # l(y | phi)


class iFilter:
    """
    An intensity filter to track multiple moving targets.
    """
    def __init__(self, rng=None):
        rng = np.random.default_rng(rng)
        self.N1 = 120000
        self.N2 = 60000
        self.N0 = self.N1 + self.N2
        self.Qinv = np.linalg.inv(Q)

        # initialise particles
        # states: x1, x2, v1, v2
        self.particles = self.init_particles(self.N0, rng=rng)

        # initialise intensities
        self.intensities = np.empty(self.N0)
        self.intensities[:self.N1] = 1. / self.N1
        self.intensities[self.N1:] = 0.
        self.null_intensity = 0.  # intensity of the null point

        # estimated number of true and false targets
        self.num_true_targets = 0.
        self.num_false_targets = 0.

    def init_particles(self, num_particles: int, rng=None) -> np.ndarray:
        """
        Initialise particles.
        """
        rng = np.random.default_rng(rng)
        N = num_particles
        particles = np.empty((N, 4))
        particles[:, :2] = rng.uniform(-800, 800, size=(N, 2))
        particles[:, 2:] = rng.uniform(-10,  10,  size=(N, 2))
        return particles

    def resample_particles(self, rng=None):
        """
        Resample particles (after measurement update).
        """
        rng = np.random.default_rng(rng)
        inten = self.intensities.sum()
        self.particles[:self.N1, :] = rng.choice(self.particles, p=self.intensities / inten, size=self.N1, replace=True)
        self.intensities[:self.N1] = inten / self.N1
        self.particles[self.N1:, :] = self.init_particles(self.N2, rng=rng)
        self.intensities[self.N1:] = 0.

    def transition_update(self, rng=None):
        """
        The motion update (a.k.a. transition update).
        """
        rng = np.random.default_rng(rng)

        # move particles as per motion model
        MU = np.dot(F, self.particles.T).T
        self.particles[:] = MU + rng.multivariate_normal(np.zeros(4), Q, size=self.N0)

        # compute intensities of particles as per Eq. (5.12)

        new_intensities = np.full(self.N0, Q_S_PHI * (1. - Q_PHI_PHI) * self.null_intensity)
        self.null_intensity = Q_PHI_S * self.intensities[:self.N1].sum() + Q_PHI_PHI * self.null_intensity
        c = 1. / (4. * np.pi * np.pi * np.sqrt(np.linalg.det(Q)))
        for n in tqdm(range(self.N0)):
            dX = self.particles[n].reshape(1, 4) - MU
            qn = c * np.exp(-0.5 * np.sum(np.dot(dX, self.Qinv) * dX, axis=1))
            qn /= (qn.sum() / (1. - Q_S_PHI))  # normalise as per Eq. (5.11)
            new_intensities[n] += (1. - Q_PHI_S) * np.dot(qn, self.intensities)
        self.intensities[:] = new_intensities

    def observation_update(self, obs: np.ndarray, rng=None):
        """
        The information update and PPP approximation (a.k.a. observation update).
        """
        assert obs.ndim == 2
        assert obs.shape[0] > 0
        rng = np.random.default_rng(rng)
        n_obs = obs.shape[0]

        # compute the integrals using particles as per the last equation
        # on page 116 (for all measurements).

        integrals = np.zeros(n_obs)
        L = np.zeros((n_obs, self.N0)) # L[r, n] is the likelihood l(y_r | s_n)
        cov = OBS_COV
        cov_inv = np.linalg.inv(cov)
        cov_det = np.linalg.det(cov)
        c1 = 1. / (2. * np.pi * np.sqrt(cov_det))
        for r in tqdm(range(n_obs)):
            dy = obs[r, :].reshape(1, -1) - self.particles[:, :2]
            lr = c1 * np.exp(-0.5 * np.sum(np.dot(dy, cov_inv) * dy, axis=1))
            L[r, :] = lr
            integrals[r] = P_D_S * np.sum(lr * self.intensities)

        # update intensities as per Eq. (5.38)

        lambda_phi_y = L_Y_PHI * P_D_PHI * self.null_intensity
        for n in tqdm(range(self.N0)):
            self.intensities[n] *= (1. - P_D_S + P_D_S * np.sum(L[:, n] / (lambda_phi_y + integrals)))

        # revise the second equation of Eq. (5.38) by moving
        # the last right brace before the first plus sign,
        # because f^{\Xi^-}(\phi) has been "absorbed" into the numerator \lambda_\phi(y).
        self.null_intensity = self.null_intensity * (1. - P_D_PHI) + np.sum(lambda_phi_y / (lambda_phi_y + integrals))

        # update the estimated number of true and false targets
        self.num_true_targets = self.intensities.sum()
        self.num_false_targets = self.null_intensity

    def expected_false_targets(self) -> float:
        """
        Estimate the expected number of false targets.
        """
        return self.num_false_targets

    def expected_true_targets(self) -> float:
        """
        Estimate the expected number of targets.
        """
        return self.num_true_targets

    def get_intensity_map(self, grid_size: int=1) -> np.ndarray:
        """
        Estimate the intensity map over the square from (-800, -800) to (800, 800).
        """
        assert grid_size > 0
        assert grid_size < 1600
        imap = np.zeros((1600 // grid_size, 1600 // grid_size))
        for i in range(self.particles.shape[0]):
            x1, x2 = self.particles[i, :2]  # note that x1 determines the column, x2 determines the row
            row_ix = math.floor((x2 + 800) / grid_size)
            col_ix = math.floor((x1 + 800) / grid_size)
            if 0 <= row_ix < imap.shape[0] and 0 <= col_ix < imap.shape[1]:
                imap[row_ix, col_ix] += self.intensities[i]  # accumulate particle intensities in the cell
        return imap

