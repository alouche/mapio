import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
from typing import Callable, Tuple, List, Optional
import time


class MAPIO:
    """
    The Multi-Modal Adaptive Path Integral Optimization (MAPIO) algorithm is a global optimization technique.
    This algorithm combines path integral methods with modal memory, adaptive step sizes,
    and information-guided exploration to efficiently locate global minima in complex landscapes.
    """

    def __init__(
        self,
        objective_func: Callable[[np.ndarray], float],
        dim: int,
        bounds: np.ndarray,
        M: int = 100,  # Number of parallel paths
        K: int = 20,  # Steps per path
        tau_0: float = 2.0,  # Initial temperature
        alpha: float = 0.8,  # Momentum blending
        beta: float = 0.85,  # Velocity decay
        hbar: float = 0.2,  # Adaptation aggressiveness
        sigma_0: float = 0.5,  # Initial step size
        delta_mode: float = 0.2,  # Mode separation threshold
        p_hop: float = 0.2,  # Basin hopping probability
        lambda_balance: float = 0.3,  # Information-exploitation balance
        gamma: float = 0.3,  # Perturbation strength
        kappa: float = 1.0,  # Density response factor
        phi: float = 0.5,  # Mode avoidance factor
        min_iter: int = 20,  # Minimum iterations
        max_iter: int = 1000,  # Maximum iterations
        epsilon_abs: float = 1e-6,  # Absolute convergence threshold
        epsilon_rel: float = 1e-4,  # Relative convergence threshold
        c_mode: int = 20,  # Mode stability threshold
        c_max: int = 50,  # Maximum stagnation count
        sigma_hop: float = 1.0,  # Step size for basin hopping
        verbose: bool = True,  # Print progress
        callback: Optional[Callable] = None,  # Callback for benchmark suite
    ):
        self.f = objective_func
        self.callback = callback
        self.dim = dim
        self.bounds = bounds
        self.lower_bounds = bounds[:, 0]
        self.upper_bounds = bounds[:, 1]

        self.M = M
        self.K = K
        self.tau_0 = tau_0
        self.alpha = alpha
        self.beta = beta
        self.hbar = hbar
        self.sigma_base = sigma_0
        self.sigma_max = (self.upper_bounds - self.lower_bounds).max() / 5
        self.delta_mode = delta_mode
        self.p_hop = p_hop * 1.5
        self.lambda_balance = lambda_balance * 1.3
        self.gamma = gamma * 1.5
        self.kappa = kappa
        self.phi = phi
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.epsilon_abs = epsilon_abs
        self.epsilon_rel = epsilon_rel
        self.c_mode = c_mode
        self.c_max = c_max
        self.sigma_hop = sigma_hop
        self.verbose = verbose

        self.modal_memory = []
        self.best_position = None
        self.best_value = float("inf")
        self.t = 0
        self.c_stale = 0
        self.mode_history = []
        self.epsilon = 1e-12
        self.delta_reset = 0.05
        self.rho = self.delta_mode / 2

        self._adapt_parameters_to_landscape()

    def _adapt_parameters_to_landscape(self, initial_samples=100):
        """Analyze landscape characteristics and adapt parameters accordingly"""
        sample_points = np.random.uniform(
            self.lower_bounds, self.upper_bounds, (initial_samples, self.dim)
        )
        sample_values = np.array([self.f(p) for p in sample_points])

        value_range = np.max(sample_values) - np.min(sample_values)
        ruggedness = np.std(sample_values) / (value_range + self.epsilon)

        is_local_min = np.array(
            [
                all(
                    sample_values[i] <= sample_values[j]
                    for j in range(initial_samples)
                    if 0 < np.sum((sample_points[i] - sample_points[j]) ** 2) < 0.1
                )
                for i in range(initial_samples)
            ]
        )
        multimodality = np.sum(is_local_min) / initial_samples

        if ruggedness > 0.7:
            self.tau_0 *= 1.5
            self.p_hop *= 1.5
            self.sigma_base *= 0.8
        elif multimodality > 0.2:
            self.lambda_balance *= 1.3
            self.gamma *= 1.3
            self.p_hop *= 1.2
        else:
            self.hbar *= 1.2
            self.sigma_base *= 0.7

    def _enforce_bounds(self, positions: np.ndarray) -> np.ndarray:
        """Enforce search domain bounds on positions."""
        return np.clip(positions, self.lower_bounds, self.upper_bounds)

    def _compute_mode_distances(self, positions: np.ndarray) -> np.ndarray:
        """Compute distances from each position to each mode."""
        mode_positions = np.array([m[0] for m in self.modal_memory])
        if len(mode_positions) == 0:
            return np.full((positions.shape[0], 1), float("inf"))

        pos_reshaped = positions.reshape(positions.shape[0], 1, self.dim)
        modes_reshaped = mode_positions.reshape(1, mode_positions.shape[0], self.dim)

        return np.sqrt(np.sum((pos_reshaped - modes_reshaped) ** 2, axis=2))

    def _compute_min_mode_distances(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute minimum distance to any mode for each position."""
        distances = self._compute_mode_distances(positions)
        min_distances = (
            distances.min(axis=1)
            if distances.shape[1] > 0
            else np.full(positions.shape[0], float("inf"))
        )
        closest_mode_indices = (
            distances.argmin(axis=1)
            if distances.shape[1] > 0
            else np.zeros(positions.shape[0], dtype=int)
        )
        return min_distances, closest_mode_indices

    def _compute_mode_density(self) -> np.ndarray:
        """Compute mode density - count of paths near each mode."""
        if not self.modal_memory:
            return np.array([])

        mode_positions = np.array([m[0] for m in self.modal_memory])
        n_modes = len(mode_positions)
        density = np.zeros(n_modes)

        if hasattr(self, "current_positions"):
            distances = self._compute_mode_distances(self.current_positions)
            density = np.sum(distances < self.rho, axis=0)

        return density

    def _update_modal_memory(self, positions: np.ndarray, values: np.ndarray) -> bool:
        """Update modal memory with newly discovered potential modes."""
        updated = False

        best_idx = np.argmin(values)
        best_candidate = positions[best_idx]
        best_candidate_value = values[best_idx]

        if best_candidate_value < self.best_value:
            self.best_position = best_candidate.copy()
            self.best_value = best_candidate_value
            updated = True
            self.c_stale = 0
        else:
            self.c_stale += 1

        if not self.modal_memory:
            self.modal_memory.append((best_candidate.copy(), best_candidate_value))
            updated = True
        else:
            min_dist = self._compute_mode_distances(best_candidate.reshape(1, -1))[
                0
            ].min()

            if min_dist > self.delta_mode:
                self.modal_memory.append((best_candidate.copy(), best_candidate_value))
                updated = True

        self.modal_memory.sort(key=lambda x: x[1])

        return updated

    def _calculate_temperature(self) -> float:
        """Improved temperature scheduling with function-specific adaptation"""
        if hasattr(self, "best_value") and hasattr(self, "prev_best_value"):
            progress_rate = abs(self.best_value - self.prev_best_value) / (
                abs(self.prev_best_value) + self.epsilon
            )

            if progress_rate < 0.001 and self.c_stale > 5:
                reheat_factor = 1.5
            else:
                reheat_factor = 1.0
        else:
            reheat_factor = 1.0

        cycle_factor = 1.0 + 0.6 * np.sin(self.t / 20) + 0.3 * np.sin(self.t / 7)
        tau_base = self.tau_0 / (1 + 0.05 * self.t) * cycle_factor * reheat_factor

        if hasattr(self, "current_positions"):
            diversity = np.mean(
                [np.var(self.current_positions[:, d]) for d in range(self.dim)]
            )
            mode_density = self._compute_mode_density()

            if len(mode_density) > 0:
                p = mode_density / (np.sum(mode_density) + self.epsilon)
                entropy = -np.sum(p * np.log(p + self.epsilon))
                normalized_entropy = entropy / np.log(len(mode_density) + self.epsilon)

                tau = tau_base * (1 + np.tanh(diversity * normalized_entropy))
            else:
                tau = tau_base * (1 + 0.5 * np.tanh(diversity))
        else:
            tau = tau_base

        return tau

    def _adapt_step_size(self, improved: bool) -> None:
        """Adapt step size based on search progress."""
        diversity = 0
        if hasattr(self, "current_positions"):
            diversity = np.mean(
                [np.var(self.current_positions[:, d]) for d in range(self.dim)]
            )

        if improved:
            self.sigma_base = self.sigma_base / (1 + self.hbar * diversity)
        else:
            self.sigma_base = min(
                self.sigma_base * (1 + self.hbar * self.c_stale), self.sigma_max
            )

    def _compute_path_weights(self, path_values: np.ndarray) -> np.ndarray:
        """Compute weights for paths using Simpson's rule for integration."""
        tau = self._calculate_temperature()

        weights = np.zeros(self.M)

        for i in range(self.M):
            if self.K % 2 == 0:
                simpson_coefs = np.ones(self.K)
                simpson_coefs[1:-1:2] = 4
                simpson_coefs[2:-1:2] = 2

                integral = np.sum(simpson_coefs * np.exp(-path_values[i] / tau)) * (
                    1.0 / 3.0
                )
            else:
                simpson_coefs = np.ones(self.K - 1)
                simpson_coefs[1:-1:2] = 4
                simpson_coefs[2:-1:2] = 2

                integral_main = np.sum(
                    simpson_coefs * np.exp(-path_values[i, :-1] / tau)
                ) * (1.0 / 3.0)
                integral_last = (
                    np.exp(-path_values[i, -2] / tau)
                    + np.exp(-path_values[i, -1] / tau)
                ) * 0.5

                integral = integral_main + integral_last

            weights[i] = integral

        norm_perf_weights = weights / (np.sum(weights) + self.epsilon)

        min_dists, _ = self._compute_min_mode_distances(self.current_positions)
        info_weights = min_dists * self._get_position_diversity()
        norm_info_weights = info_weights / (np.sum(info_weights) + self.epsilon)

        if self.c_stale > 10:
            adaptive_lambda = min(0.8, self.lambda_balance * (1 + 0.1 * self.c_stale))
        else:
            adaptive_lambda = self.lambda_balance

        blended_weights = (
            1 - adaptive_lambda
        ) * norm_perf_weights + adaptive_lambda * norm_info_weights

        return blended_weights / np.sum(blended_weights)

    def _get_position_diversity(self):
        """Helper method to compute position diversity."""
        position_diversity = np.zeros(self.M)
        for i in range(self.M):
            dists = np.sqrt(
                np.sum(
                    (self.current_positions - self.current_positions[i]) ** 2, axis=1
                )
            )
            nearby = np.sum(dists < 0.1 * self.delta_mode)
            position_diversity[i] = 1.0 / (nearby + 1)
        return position_diversity

    def _apply_diversity_enforcement(self) -> None:
        """Apply mode-aware diversity enforcement to prevent over-concentration."""
        if len(self.modal_memory) == 0 or not hasattr(self, "current_positions"):
            return

        k = max(1, self.M // 10)
        _, labels = kmeans2(self.current_positions, k, minit="points")

        cluster_counts = np.bincount(labels, minlength=k)
        if np.any(cluster_counts == 0):
            _, labels = kmeans2(self.current_positions, k, minit="random")
            cluster_counts = np.bincount(labels, minlength=k)

        dense_threshold = np.mean(cluster_counts) * 1.5
        dense_clusters = np.where(cluster_counts > dense_threshold)[0]

        if len(dense_clusters) == 0:
            return

        for cluster_idx in dense_clusters:
            path_indices = np.where(labels == cluster_idx)[0]

            _, closest_modes = self._compute_min_mode_distances(
                self.current_positions[path_indices]
            )

            for i, path_idx in enumerate(path_indices):
                if closest_modes[i] < len(self.modal_memory):
                    mode_idx = closest_modes[i]
                    mode_density = self._compute_mode_density()[mode_idx]
                    perturbation_strength = (
                        self.gamma * mode_density * (1 + 0.2 * self.t / self.max_iter)
                    )
                    perturbation = np.random.normal(
                        0, perturbation_strength * self.sigma_base, self.dim
                    )
                    self.current_positions[path_idx] += perturbation

        self.current_positions = self._enforce_bounds(self.current_positions)

    def _voronoi_allocation(self) -> List[int]:
        """Allocate paths to regions based on Voronoi cells."""
        if len(self.modal_memory) <= 1:
            return [0] * self.M

        mode_values = np.array([m[1] for m in self.modal_memory])

        min_dists, closest_modes = self._compute_min_mode_distances(
            np.random.uniform(
                self.lower_bounds, self.upper_bounds, size=(1000, self.dim)
            )
        )
        volumes = np.bincount(closest_modes, minlength=len(self.modal_memory))
        volumes = volumes / np.sum(volumes)

        tau = self._calculate_temperature()

        weights = np.exp(-(mode_values - np.min(mode_values)) / tau) * volumes
        weights = weights / np.sum(weights)

        path_allocations = []
        remaining = self.M

        for i in range(len(weights)):
            n_i = int(np.floor(self.M * weights[i]))
            path_allocations.extend([i] * min(n_i, remaining))
            remaining -= n_i

        best_mode_idx = np.argmin(mode_values)
        path_allocations.extend([best_mode_idx] * remaining)

        assert len(path_allocations) == self.M
        return path_allocations

    def _periodic_readapt_parameters(self):
        """
        Periodically re-analyze landscape characteristics during optimization
        based on collected data and adapt parameters accordingly.
        """
        if self.t % 50 != 0 or self.t == 0:
            return False

        if not hasattr(self, "current_positions"):
            return False

        sample_points = self.current_positions.copy()
        sample_values = np.array([self.f(p) for p in sample_points])

        if self.modal_memory:
            mode_positions = np.array([m[0] for m in self.modal_memory])
            mode_values = np.array([m[1] for m in self.modal_memory])
            sample_points = np.vstack([sample_points, mode_positions])
            sample_values = np.append(sample_values, mode_values)

        value_range = np.max(sample_values) - np.min(sample_values)
        ruggedness = np.std(sample_values) / (value_range + self.epsilon)

        if len(sample_points) > 5:
            Z = linkage(sample_points, "average")
            cluster_threshold = 0.1 * np.max(Z[:, 2])
            clusters = fcluster(Z, cluster_threshold, criterion="distance")
            n_clusters = len(np.unique(clusters))

            multimodality = n_clusters / len(sample_points)

            centers = np.zeros((n_clusters, self.dim))
            for i in range(n_clusters):
                cluster_indices = np.where(clusters == i + 1)[0]
                if len(cluster_indices) > 0:
                    centers[i] = np.mean(sample_points[cluster_indices], axis=0)

            if n_clusters > 1:
                center_distances = pdist(centers)
                avg_separation = np.mean(center_distances)
                domain_size = np.sqrt(
                    np.sum((self.upper_bounds - self.lower_bounds) ** 2)
                )
                normalized_separation = avg_separation / domain_size
            else:
                normalized_separation = 0.0
        else:
            multimodality = 0.0
            normalized_separation = 0.0

        if hasattr(self, "value_history") and len(self.value_history) > 10:
            recent_values = np.array(self.value_history[-10:])
            convergence_rate = np.abs(
                (recent_values[0] - recent_values[-1])
                / (recent_values[0] + self.epsilon)
            )
        else:
            convergence_rate = 0.1

        if not hasattr(self, "value_history"):
            self.value_history = []
        self.value_history.append(self.best_value)

        if self.verbose:
            print(f"\nLandscape analysis at iteration {self.t}:")
            print(f"  Ruggedness: {ruggedness:.4f}")
            print(f"  Multimodality: {multimodality:.4f}")
            print(f"  Mode separation: {normalized_separation:.4f}")
            print(f"  Convergence rate: {convergence_rate:.4f}")

        params_updated = False

        if ruggedness > 0.6 and convergence_rate < 0.01:
            old_tau = self.tau_0
            self.tau_0 = min(self.tau_0 * 1.5, 10.0)
            params_updated = params_updated or (old_tau != self.tau_0)
        elif ruggedness < 0.2 and convergence_rate < 0.01:
            old_tau = self.tau_0
            self.tau_0 = max(self.tau_0 * 0.8, 0.5)
            params_updated = params_updated or (old_tau != self.tau_0)

        if multimodality > 0.3:
            old_p_hop = self.p_hop
            old_gamma = self.gamma
            self.p_hop = min(self.p_hop * 1.2, 0.5)
            self.gamma = min(self.gamma * 1.2, 0.8)
            params_updated = params_updated or (
                old_p_hop != self.p_hop or old_gamma != self.gamma
            )
        elif multimodality < 0.1 and len(self.modal_memory) < 3:
            old_p_hop = self.p_hop
            old_gamma = self.gamma
            self.p_hop = min(self.p_hop * 1.5, 0.6)
            self.gamma = min(self.gamma * 1.3, 0.8)
            params_updated = params_updated or (
                old_p_hop != self.p_hop or old_gamma != self.gamma
            )

        if normalized_separation > 0.5:
            old_sigma = self.sigma_base
            self.sigma_base = min(self.sigma_base * 1.3, self.sigma_max)
            params_updated = params_updated or (old_sigma != self.sigma_base)
        elif normalized_separation < 0.1 and convergence_rate < 0.02:
            old_sigma = self.sigma_base
            old_hbar = self.hbar
            self.sigma_base = max(self.sigma_base * 0.8, 0.01)
            self.hbar = min(self.hbar * 1.2, 0.5)
            params_updated = params_updated or (
                old_sigma != self.sigma_base or old_hbar != self.hbar
            )

        if self.c_stale > 5 and convergence_rate < 0.01:
            old_lambda = self.lambda_balance
            self.lambda_balance = min(self.lambda_balance * 1.3, 0.7)
            params_updated = params_updated or (old_lambda != self.lambda_balance)
        elif convergence_rate > 0.1:
            old_lambda = self.lambda_balance
            self.lambda_balance = max(self.lambda_balance * 0.9, 0.1)
            params_updated = params_updated or (old_lambda != self.lambda_balance)

        if params_updated and self.verbose:
            print("  Parameters adapted based on landscape analysis:")
            print(f"    tau_0: {self.tau_0:.4f}")
            print(f"    p_hop: {self.p_hop:.4f}")
            print(f"    gamma: {self.gamma:.4f}")
            print(f"    sigma_base: {self.sigma_base:.4f}")
            print(f"    lambda_balance: {self.lambda_balance:.4f}")
            print(f"    hbar: {self.hbar:.4f}")

        return params_updated

    def _initialize_paths(self) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize paths using modal memory, basin hopping, and origin bias."""
        positions = np.zeros((self.M, self.dim))
        velocities = np.zeros((self.M, self.dim))

        origin_bias = np.random.random(self.M) < 0.35
        positions[origin_bias] = np.random.normal(
            0, 0.2, (np.sum(origin_bias), self.dim)
        )

        non_biased = ~origin_bias
        remaining = np.sum(non_biased)

        if not self.modal_memory:
            positions[non_biased] = np.random.uniform(
                self.lower_bounds, self.upper_bounds, size=(remaining, self.dim)
            )
        else:
            allocations = self._voronoi_allocation()
            mode_positions = np.array([m[0] for m in self.modal_memory])

            hop_mask = np.random.random(remaining) < (
                self.p_hop * (1 + 0.5 * self.c_stale / self.c_max)
            )

            hop_count = 0
            alloc_count = 0
            for i in range(self.M):
                if origin_bias[i]:
                    continue

                if hop_mask[hop_count]:
                    mode_idx = np.random.choice(len(self.modal_memory))
                    positions[i] = mode_positions[mode_idx] + np.random.normal(
                        0, self.sigma_hop, self.dim
                    )
                    hop_count += 1
                else:
                    mode_idx = allocations[alloc_count % len(allocations)]
                    alloc_count += 1
                    positions[i] = mode_positions[mode_idx] + np.random.normal(
                        0, self.sigma_base * 0.1, self.dim
                    )

            best_mask = np.random.random(self.M) < 0.15  # TODO: 15% of paths
            if self.best_position is not None:
                positions[best_mask] = self.best_position + np.random.normal(
                    0, self.sigma_base * 0.01, (np.sum(best_mask), self.dim)
                )

        positions = self._enforce_bounds(positions)

        return positions, velocities

    def _generate_paths(
        self, positions: np.ndarray, velocities: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate exploration paths and evaluate the objective function."""
        path_positions = np.zeros((self.M, self.K, self.dim))
        path_velocities = np.zeros((self.M, self.K, self.dim))
        path_values = np.zeros((self.M, self.K))

        path_positions[:, 0, :] = positions
        path_velocities[:, 0, :] = velocities

        for i in range(self.M):
            path_values[i, 0] = self.f(positions[i])

        for k in range(1, self.K):
            if self.modal_memory:
                min_dists, closest_modes = self._compute_min_mode_distances(
                    path_positions[:, k - 1, :]
                )
                mode_density = self._compute_mode_density()

                sigma_i = np.zeros(self.M)
                for i in range(self.M):
                    if closest_modes[i] < len(mode_density):
                        sigma_i[i] = self.sigma_base / (
                            1 + self.kappa * mode_density[closest_modes[i]]
                        )
                    else:
                        sigma_i[i] = self.sigma_base
            else:
                sigma_i = np.full(self.M, self.sigma_base)

            if self.t % 5 == 0:
                reset_indices = np.arange(self.M)
            else:
                min_dists, _ = self._compute_min_mode_distances(
                    path_positions[:, k - 1, :]
                )
                reset_indices = np.where(min_dists < self.delta_reset)[0]

            if len(reset_indices) > 0:
                path_velocities[reset_indices, k - 1, :] = 0

            random_steps = np.random.normal(0, 1, (self.M, self.dim))
            for i in range(self.M):
                random_steps[i] *= sigma_i[i]

            path_velocities[:, k, :] = (
                self.beta * path_velocities[:, k - 1, :]
                + (1 - self.beta) * random_steps
            )

            path_positions[:, k, :] = (
                path_positions[:, k - 1, :]
                + self.alpha * path_velocities[:, k - 1, :]
                + random_steps
            )

            path_positions[:, k, :] = self._enforce_bounds(path_positions[:, k, :])

            for i in range(self.M):
                path_values[i, k] = self.f(path_positions[:, k, :][i])

        return path_positions, path_velocities, path_values

    def _gradient_refinement(self, positions: np.ndarray) -> np.ndarray:
        """Enhanced gradient-based refinement"""
        refined_positions = positions.copy()

        for i in range(positions.shape[0]):
            pos = positions[i].copy()
            f_pos = self.f(pos)

            if self.c_stale > 10:
                h = self.sigma_base * 0.001
            else:
                h = self.sigma_base * 0.01

            grad = np.zeros(self.dim)
            for j in range(self.dim):
                pos_plus = pos.copy()
                pos_plus[j] += h
                f_plus = self.f(pos_plus)

                pos_minus = pos.copy()
                pos_minus[j] -= h
                f_minus = self.f(pos_minus)

                grad[j] = (f_plus - f_minus) / (2 * h)

            grad_norm = np.linalg.norm(grad)
            if grad_norm > self.epsilon:
                grad = grad / grad_norm

                best_step = 0.0
                best_value = f_pos

                for step_factor in [0.01, 0.05, 0.1, 0.2, 0.5]:
                    trial_pos = pos - step_factor * self.sigma_base * grad
                    trial_pos = np.clip(trial_pos, self.lower_bounds, self.upper_bounds)
                    trial_value = self.f(trial_pos)

                    if trial_value < best_value:
                        best_value = trial_value
                        best_step = step_factor

                refined_positions[i] = pos - best_step * self.sigma_base * grad

        return self._enforce_bounds(refined_positions)

    def _final_refinement(self, candidates, n_iterations=50):
        """
        Apply BFGS local refinement to the best candidates
        """

        refined_positions = []
        refined_values = []

        for pos in candidates:
            bounds = [
                (self.lower_bounds[i], self.upper_bounds[i]) for i in range(self.dim)
            ]

            result = minimize(
                self.f,
                pos,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": n_iterations, "disp": False},
            )

            refined_positions.append(result.x)
            refined_values.append(result.fun)

        best_idx = np.argmin(refined_values)
        return refined_positions[best_idx], refined_values[best_idx]

    def _check_termination(self) -> bool:
        """Check termination criteria."""
        if self.t < self.min_iter:
            return False

        if self.t >= self.max_iter:
            if self.verbose:
                print("Termination: Maximum iterations reached")
            return True

        if self.c_stale > self.c_max:
            if self.verbose:
                print("Termination: Stagnation detected")
            return True

        if len(self.mode_history) >= self.c_mode:
            if all(
                m == self.mode_history[-self.c_mode]
                for m in self.mode_history[-self.c_mode :]
            ):
                if self.verbose:
                    print("Termination: Mode stability achieved")
                return True

        if self.t > 0 and hasattr(self, "prev_best_value"):
            abs_diff = abs(self.prev_best_value - self.best_value)
            rel_diff = abs_diff / (abs(self.prev_best_value) + self.epsilon)

            if abs_diff < self.epsilon_abs and rel_diff < self.epsilon_rel:
                if self.verbose:
                    print("Termination: Convergence criteria met")
                return True

        return False

    def optimize(
        self, x0: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, List[Tuple[np.ndarray, float]]]:
        """
        Run the optimizer.

        Args:
            x0: Initial position (optional). If None, a random position is used.

        Returns:
            Tuple of (best_position, best_value, modal_memory)
        """
        start_time = time.time()

        if x0 is None:
            x0 = np.random.uniform(self.lower_bounds, self.upper_bounds)

        initial_value = self.f(x0)
        self.best_position = x0.copy()
        self.best_value = initial_value
        self.modal_memory.append((x0.copy(), initial_value))

        velocities = np.zeros((self.M, self.dim))

        if not hasattr(self, "value_history"):
            self.value_history = []

        self._adapt_parameters_to_landscape(initial_samples=100)

        for t in range(self.max_iter):
            self.t = t
            self.prev_best_value = self.best_value

            self.mode_history.append(len(self.modal_memory))
            positions, velocities = self._initialize_paths()
            path_positions, path_velocities, path_values = self._generate_paths(
                positions, velocities
            )

            self.current_positions = path_positions[:, -1, :]
            current_values = path_values[:, -1]
            improved = self._update_modal_memory(self.current_positions, current_values)

            self._periodic_readapt_parameters()

            self._apply_diversity_enforcement()
            refined_positions = self._gradient_refinement(self.current_positions)
            refined_values = np.array([self.f(pos) for pos in refined_positions])
            improved = (
                self._update_modal_memory(refined_positions, refined_values) or improved
            )

            self._adapt_step_size(improved)
            weights = self._compute_path_weights(path_values)
            weighted_positions = np.zeros((self.M, self.dim))
            weighted_velocities = np.zeros((self.M, self.dim))

            for i in range(self.M):
                weighted_positions[i] = np.sum(
                    weights[i] * path_positions[i, :, :], axis=0
                )
                weighted_velocities[i] = np.sum(
                    weights[i] * path_velocities[i, :, :], axis=0
                )

            positions = weighted_positions
            velocities = weighted_velocities

            if self._check_termination():
                break

            if self.verbose and (t % 10 == 0 or t == self.max_iter - 1):
                print(
                    f"Iteration {t}: Best value = {self.best_value:.6e}, Modes = {len(self.modal_memory)}, "
                    f"Sigma = {self.sigma_base:.6e}, Temp = {self._calculate_temperature():.6e}"
                )

            if self.callback is not None:
                self.callback(
                    iteration=t,
                    position=self.best_position.copy(),
                    value=self.best_value,
                    mode_positions=self.modes if hasattr(self, "modes") else None,
                )

        elapsed_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization completed in {elapsed_time:.2f} seconds")
            print(f"Final best value: {self.best_value:.10e}")
            print(f"Number of discovered modes: {len(self.modal_memory)}")
            print(
                f"Number of function evaluations: {self.t * self.M * self.K + self.t * self.M * 2 * self.dim + 1}"
            )

        if len(self.modal_memory) > 0:
            top_candidates = [
                m[0] for m in sorted(self.modal_memory, key=lambda x: x[1])[:3]
            ]
            refined_position, refined_value = self._final_refinement(top_candidates)

            if refined_value < self.best_value:
                self.best_position = refined_position
                self.best_value = refined_value

        return self.best_position, self.best_value, self.modal_memory
