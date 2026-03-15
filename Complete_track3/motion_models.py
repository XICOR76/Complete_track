"""
Approach 3 — Motion Models
Interchangeable motion models for research-oriented tracker.
  • KalmanMotionModel   — classic linear motion prediction
  • ParticleFilterModel — non-parametric, handles non-linear motion
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────────────────────────────────────

class MotionModel(ABC):
    """Interface that all motion models must implement."""

    @abstractmethod
    def initiate(self, measurement: np.ndarray) -> dict:
        """Create initial state from first measurement."""

    @abstractmethod
    def predict(self, state: dict) -> dict:
        """Advance state by one time step."""

    @abstractmethod
    def update(self, state: dict, measurement: np.ndarray) -> dict:
        """Incorporate new measurement into state."""

    @abstractmethod
    def get_bbox_estimate(self, state: dict) -> np.ndarray:
        """Return [x1, y1, x2, y2] from state."""


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter Motion Model
# ─────────────────────────────────────────────────────────────────────────────

class KalmanMotionModel(MotionModel):
    """
    8-dimensional constant-velocity Kalman filter.
    State: [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement: [cx, cy, w, h]

    Suitable for: linear, predictable motion (walking pedestrians, vehicles)
    """

    def __init__(
        self,
        process_noise_std: float = 1.0,
        measurement_noise_std: float = 1.0,
        velocity_uncertainty: float = 10.0
    ):
        dt = 1.0

        # State transition
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Observation matrix
        self.H = np.eye(4, 8)

        self.q = process_noise_std
        self.r = measurement_noise_std
        self.v0 = velocity_uncertainty

    def initiate(self, measurement: np.ndarray) -> dict:
        """measurement: [cx, cy, w, h]"""
        mean = np.zeros(8)
        mean[:4] = measurement

        std = np.array([
            2 * self.r * measurement[2],   # cx uncertainty ∝ width
            2 * self.r * measurement[3],   # cy uncertainty ∝ height
            1e-2,                           # w
            2 * self.r * measurement[3],   # h
            self.v0 * measurement[2],
            self.v0 * measurement[3],
            1e-5,
            self.v0 * measurement[3]
        ])
        covariance = np.diag(std ** 2)
        return {'mean': mean, 'covariance': covariance, 'type': 'kalman'}

    def predict(self, state: dict) -> dict:
        mean, cov = state['mean'], state['covariance']

        # Scale noise by bounding box size
        std_p = self.q * np.array([
            mean[2], mean[3], 1e-2, mean[3],
            mean[2], mean[3], 1e-5, mean[3]
        ])
        Q = np.diag(std_p ** 2)

        new_mean = self.F @ mean
        new_cov  = self.F @ cov @ self.F.T + Q
        return {'mean': new_mean, 'covariance': new_cov, 'type': 'kalman'}

    def update(self, state: dict, measurement: np.ndarray) -> dict:
        mean, cov = state['mean'], state['covariance']

        std_r = self.r * np.array([mean[2], mean[3], 1e-1, mean[3]])
        R = np.diag(std_r ** 2)

        S = self.H @ cov @ self.H.T + R
        K = cov @ self.H.T @ np.linalg.inv(S)

        new_mean = mean + K @ (measurement - self.H @ mean)
        new_cov  = (np.eye(8) - K @ self.H) @ cov
        return {'mean': new_mean, 'covariance': new_cov, 'type': 'kalman'}

    def get_bbox_estimate(self, state: dict) -> np.ndarray:
        cx, cy, w, h = state['mean'][:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def mahalanobis_distance(self, state: dict, measurement: np.ndarray) -> float:
        """Gating distance for cascade matching."""
        mean, cov = state['mean'], state['covariance']
        std_r = self.r * np.array([mean[2], mean[3], 1e-1, mean[3]])
        S = self.H @ cov @ self.H.T + np.diag(std_r ** 2)
        diff = measurement - self.H @ mean
        return float(diff.T @ np.linalg.inv(S) @ diff)


# ─────────────────────────────────────────────────────────────────────────────
# Particle Filter Motion Model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParticleFilterConfig:
    n_particles: int       = 300     # more particles = more accurate but slower
    process_std: float     = 10.0    # spread of state transition noise
    measurement_std: float = 20.0    # how tightly observations are weighted
    resample_threshold: float = 0.5  # effective sample size ratio for resampling


class ParticleFilterModel(MotionModel):
    """
    Sequential Monte Carlo (Particle Filter) motion model.

    Particles: each particle = [cx, cy, w, h, vcx, vcy]
    Weights:   likelihood of particle given measurement

    Advantage over Kalman:
      ✓ Handles non-linear, multi-modal distributions
      ✓ Robust to sudden motion changes (e.g., sports tracking)
      ✓ Can represent multiple hypotheses (occluded reappearance)

    Disadvantage:
      ✗ Heavier computation (N particles per track)
      ✗ Can suffer from particle degeneracy
    """

    def __init__(self, config: Optional[ParticleFilterConfig] = None):
        self.cfg = config or ParticleFilterConfig()
        self.rng = np.random.default_rng(42)

    def initiate(self, measurement: np.ndarray) -> dict:
        """
        Initialise particle cloud around first measurement.
        measurement: [cx, cy, w, h]
        """
        cx, cy, w, h = measurement
        N = self.cfg.n_particles

        # Spread initial particles with small Gaussian noise
        particles = np.zeros((N, 6))
        particles[:, 0] = cx + self.rng.normal(0, 2.0, N)    # cx
        particles[:, 1] = cy + self.rng.normal(0, 2.0, N)    # cy
        particles[:, 2] = w  + self.rng.normal(0, 1.0, N)    # w
        particles[:, 3] = h  + self.rng.normal(0, 1.0, N)    # h
        particles[:, 4] = self.rng.normal(0, 1.0, N)          # vcx
        particles[:, 5] = self.rng.normal(0, 1.0, N)          # vcy

        weights = np.ones(N) / N
        return {'particles': particles, 'weights': weights, 'type': 'particle'}

    def predict(self, state: dict) -> dict:
        """
        Apply state transition (constant velocity + Gaussian noise).
        """
        particles = state['particles'].copy()
        N = len(particles)
        std = self.cfg.process_std

        # Update positions with velocity
        particles[:, 0] += particles[:, 4] + self.rng.normal(0, std * 0.5, N)
        particles[:, 1] += particles[:, 5] + self.rng.normal(0, std * 0.5, N)
        particles[:, 2] += self.rng.normal(0, std * 0.1, N)   # w drifts slowly
        particles[:, 3] += self.rng.normal(0, std * 0.1, N)   # h drifts slowly
        particles[:, 4] += self.rng.normal(0, std * 0.3, N)   # velocity noise
        particles[:, 5] += self.rng.normal(0, std * 0.3, N)

        # Keep sizes positive
        particles[:, 2] = np.abs(particles[:, 2]).clip(5, None)
        particles[:, 3] = np.abs(particles[:, 3]).clip(5, None)

        return {'particles': particles, 'weights': state['weights'].copy(), 'type': 'particle'}

    def update(self, state: dict, measurement: np.ndarray) -> dict:
        """
        Weight particles by Gaussian likelihood of measurement.
        Then resample if effective sample size drops too low.
        """
        particles = state['particles']
        weights   = state['weights'].copy()
        cx, cy, w, h = measurement
        std = self.cfg.measurement_std

        # Likelihood: Gaussian around each dimension
        dx = (particles[:, 0] - cx) / std
        dy = (particles[:, 1] - cy) / std
        dw = (particles[:, 2] - w)  / (std * 0.5)
        dh = (particles[:, 3] - h)  / (std * 0.5)

        likelihood = np.exp(-0.5 * (dx**2 + dy**2 + dw**2 + dh**2))
        weights *= likelihood
        weights += 1e-300          # avoid degeneracy
        weights /= weights.sum()   # normalise

        # Systematic resampling when effective N falls below threshold
        N_eff = 1.0 / np.sum(weights ** 2)
        if N_eff < self.cfg.resample_threshold * len(particles):
            particles, weights = self._systematic_resample(particles, weights)

        return {'particles': particles, 'weights': weights, 'type': 'particle'}

    def get_bbox_estimate(self, state: dict) -> np.ndarray:
        """Weighted mean of particles → [x1, y1, x2, y2]."""
        particles = state['particles']
        weights   = state['weights']

        cx = np.sum(weights * particles[:, 0])
        cy = np.sum(weights * particles[:, 1])
        w  = np.sum(weights * particles[:, 2])
        h  = np.sum(weights * particles[:, 3])

        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])

    def get_uncertainty(self, state: dict) -> np.ndarray:
        """Weighted variance of particles — useful for adaptive gating."""
        particles = state['particles']
        weights   = state['weights']

        mean = np.average(particles[:, :4], weights=weights, axis=0)
        var  = np.average((particles[:, :4] - mean) ** 2, weights=weights, axis=0)
        return var   # [var_cx, var_cy, var_w, var_h]

    def _systematic_resample(
        self,
        particles: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Systematic resampling — O(N), less variance than multinomial.
        """
        N = len(particles)
        positions = (self.rng.random() + np.arange(N)) / N
        indices = np.searchsorted(np.cumsum(weights), positions)
        indices = np.clip(indices, 0, N - 1)

        new_particles = particles[indices]
        new_weights   = np.ones(N) / N
        return new_particles, new_weights
