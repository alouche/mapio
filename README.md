# MAPIO

The Multi-Modal Adaptive Path Integral Optimization (MAPIO) algorithm is a global optimization technique combining stochastic path integration (inspired by Feynman's path integral formulation in Quantum mechanics) with adaptive landscape exploration. It employs parallel path evaluation, modal memory for local minima tracking, and hybrid stochastic-deterministic refinement.

MAPIO specializes in navigating complex, high-dimensional, non-convex spaces through:

- Hybrid Stochastic-Entropic Dynamics: Combines Fokker-Planck evolution with Voronoi-based mass transport
- Non-Markovian Temperature Control: Cyclic reheating schedule to prevent ergodicity breaking
- Geometric Parameter Adaptation: Direct coupling between landscape curvature metrics and exploration parameters

To learn more on how the algorithm came to be, click on [Inspiration from Physics](docs/inspiration_from_physics.md). In essence, MAPIO synthesizes concepts from two fundamental physical phenomena:

- Quantum mechanics principle where particles explore all possible paths simultaneously, with probabilities weighted by their action (Feynman, 1948)
- Photons naturally find optimal paths through interference, as seen in Fermat's principle of least time

| Concept               | Mathematical Representation                                                                                                             | Physical Analogy              |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| Path Position        | ![\mathbf{x}_i^{(k)} \in \mathbb{R}^d](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\mathbf{x}_i^{(k)}%20\in%20\mathbb{R}^d)         | Quantum particle trajectory   |
| Path Velocity        | ![\mathbf{v}_i^{(k)} = \beta \mathbf{v}_i^{(k-1)} + (1 - \beta) \boldsymbol{\epsilon}_i^{(k)}](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\mathbf{v}_i^{(k)}%20=%20\beta%20\mathbf{v}_i^{(k-1)}%20+%20(1%20-%20\beta)%20\boldsymbol{\epsilon}_i^{(k)}) | Momentum in field             |
| Temperature Schedule | ![\tau(t) = \frac{\tau_0}{\sqrt{1 + t/T_0}} \cdot \left(1 + A \sin(2\pi t / T_1)\right)](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\tau(t)%20=%20\frac{\tau_0}{\sqrt{1%20+%20t/T_0}}%20\cdot%20\left(1%20+%20A%20\sin(2\pi%20t%20/%20T_1)\right)) | Thermal annealing             |
| Mode Density         | ![\rho_j = \frac{1}{\pi r^2} \sum_{k=1}^{M} e^{-\|\mathbf{x}_k - \mathbf{m}_j\|^2 / 2r^2}](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}\rho_j%20=%20\frac{1}{\pi%20r^2}%20\sum_{k=1}^{M}%20e^{-\|\mathbf{x}_k%20-%20\mathbf{m}_j\|^2%20/%202r^2}) | Photon concentration          |

For the extensive mathematical formulations click here [Mathematical Foundation](docs/mathematical_foundation.md).

The strength of MAPIO lies in simultaneous global exploration and gradient-aware local refinement, making it particularly effective for:

- Noisy/rugged objective functions
- Problems requiring multiple solution candidates
- Landscapes with deceptive local minima

The empirical advantages of MAPI over current state-of-the-art algorithms are:

1. **Mode Discovery Rate**: 
![O(M^{2/3})](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}O(M^{2/3})) vs ![O(M^{1/2})](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}O(M^{1/2})) in standard GA

2. **Convergence Speed**: 38% faster on Rastrigin vs. CMA-ES (d=10)

3. **Parameter Sensitivity**:  Normalized sensitivity  ![S_p = \frac{\partial \mathcal{V}}{\partial p} \cdot \frac{p}{\mathcal{V}} < 0.2](https://latex.codecogs.com/png.latex?\bg{FFFFFF}\fg{000000}S_p%20=%20\frac{\partial%20\mathcal{V}}{\partial%20p}%20\cdot%20\frac{p}{\mathcal{V}}%20<%200.2) for all parameters.

4. **High Parallelism**: MAPIO's algorithm allows it to simultaneously explore multiple regions, making it particularly well-suited for distributed computing environments.


## Benchmarks

Across 5 and 10 dimensions, MAPIO was bencharmked without fine tuning againt the following optimization algorithms:Dual Annealing, Differential Evolution and the current state of the art algorithm CMA-ES.

Acoss the various benchmarks, MAPIO consistently delivered near-optimal solutions with robust mode exploration, while CMA-ES, designed for rapid convergence in smoother, unimodal landscapes, predictably struggles in these highly multimodal topologies by often converging prematurely to local optima. Differential Evolution and Dual Annealing showed competitive performance but sometimes settled for local solutions, making them less reliable overall. These findings hints that MAPIO excels in complex optimization landscapes compared to current state of the art algorithms.

Specifically, MAPIO demonstrated:

- Reliable global convergence on challenging and deceptive landscapes
- Explicity mode discovery, facilitating detailed analysis of the solution space
- Adaptive resource allocation by effectively balancing exploration and exploitation


The topologies tested were:

| **Function**        | **Description**                                                                                                                                                                                                                     | **Results**                                                                                                     |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| **Rastrigin**        | Highly multimodal with regular local minima and a single global minimum at the origin.                                                                                                      | [5D](results/rastrigin-5d_report.md), [10D](results/rastrigin-10d_report.md)                                   |
| **Rosenbrock**       | Also known as the "banana function," it is a unimodal test function characterized by a narrow, curved valley leading to the global minimum at (1,1), where f(x,y)=0, making convergence challenging despite its simple structure. | [5D](results/rosenbrock-5d_report.md), [10D](results/rosenbrock-10d_report.md)                 |
| **Ackley**           | A nearly flat outer region with many local minima and a deep global minimum at the origin, making it challenging for optimization algorithms due to its deceptive landscape.                  | [5D](results/ackley-5d_report.md), [10D](results/ackley-10d_report.md)                         |
| **Levy**             | A multimodal landscape with a global minimum at (1,1,...,1), making it challenging for optimization due to its complex, non-separable structure.                                            | [5D](results/levy-5d_report.md), [10D](results/levy-10d_report.md)                             |
| **Schwefel**         | A complex landscape with numerous local minima and a global minimum far from the center.                                                                                                   | [5D](results/schwefel-5d_report.md), [10D](results/schwefel-10d_report.md)                     |
| **Griewank**         | A smooth, multimodal landscape with regularly distributed local minima and a global minimum at the origin, making it challenging yet structured for optimization.                           | [5D](results/griewank-5d_report.md), [10D](results/griewank-10d_report.md)                     |
| **Michalewicz**      | A complex multimodal landscape with n! local minima, steep valleys and ridges, and a global minimum that becomes increasingly difficult to locate as the number of dimensions increases.     | [5D](results/michalewicz-5d_report.md), [10D](results/michalewicz-10d_report.md)               |
| **Styblinski-Tang**  | A multimodal landscape with a global minimum at xi=−2.903534 and f(x∗)=−39.16599n, typically evaluated in the domain [-5, 5].                                                               | [5D](results/styblinski_tang-5d_report.md), [10D](results/styblinski_tang-10d_report.md)       |

## Applications

Due to its inherent capability to balance exploitation and exploration, MAPIO positions itself as a powerful tool for navigating complex optimization problems with multi-modal and deceptive landscape. Specifically in the areas of:


**Neural Architecture Search (NAS)**:

MAPIO's ability to explore thoroughly can help uncover unique neural network designs by efficiently navigating the design space. This could lead to discovering architectures that traditional methods might overlook.

**Adversarial Robustness**:

MAPIO might also be useful for creating or defending against adversarial examples. Its strength in navigating complex optimization landscapes makes it valuable for identifying robust perturbations

**Training Complex Models**:

- Generative Adversarial Networks (GANs): MAPIO could help balance the competing objectives of GANs, reducing problems like mode collapse by exploring a variety of possible solutions.
- Reinforcement Learning (RL): MAPIO's ability to avoid premature convergence makes it promising for optimizing policies in environments with noisy, multimodal reward functions.

**Hyperparameter Optimization**:

MAPIO’s ability to perform a strong global search makes it ideal for tuning hyperparameters in complicated models. This is especially helpful when the optimization landscape has many peaks and valleys, where traditional methods might get stuck.

**Feature Selection and Dimensionality Reduction**:

MAPIO’s optimization skills can be applied to selecting the most important features or reducing dimensions in datasets. This can lead to better model performance and make high-dimensional data easier to understand.

## Algorithms Steps
### 1. Initialization
```python
Input: Objective function f, search space bounds, algorithm parameters
Output: Initialized optimizer state

1.1. Initialize population:
    positions[i] ~ U(lower_bounds, upper_bounds) ∀i ∈ {1,...,M}
    velocities[i] ← 0

1.2. Perform initial landscape analysis:
    Sample N points → estimate ruggedness (σ_f/Δf) and multimodality (N_modes/N)
    Adapt parameters: τ₀, σ_base, λ_balance based on landscape characteristics

1.3. Initialize modal memory with best initial sample
```

### 2. Main Optimization Loop

```
Repeat until termination:
    2.1. Path Generation:
        For each path i ∈ {1,...,M}:
            x_i^{k+1} = x_i^k + αv_i^k + βε_i^k
            v_i^{k+1} = βv_i^k + (1-β)ε_i^k
            where ε_i^k ~ N(0,σ_i^k)
        
    2.2. Mode Management:
        Update modal memory with new candidates
        Maintain sorted list of (position, value) pairs
        Add new modes if distance > δ_mode
        
    2.3. Information-Guided Adaptation:
        Compute Voronoi allocation weights:
            w_j ∝ exp(-(f_j - f_min)/τ) * V_j
            where V_j = Voronoi cell volume
        
    2.4. Diversity Enforcement:
        Apply k-means clustering (k = M/10)
        Perturb paths in dense clusters:
            x_i ← x_i + γ·N(0,σ_base)·mode_density
        
    2.5. Gradient Refinement:
        Compute finite-difference gradients
        Perform line search along descent direction
```

### 3. Termination
```
Check convergence criteria:
    - Absolute: |f_{t} - f_{t-1}| < ε_abs
    - Relative: |f_{t} - f_{t-1}|/|f_{t}| < ε_rel
    - Stagnation: c_stale > c_max
    - Mode stability: |modes_t - modes_{t-k}| = 0 ∀k ∈ [1,c_mode]
```