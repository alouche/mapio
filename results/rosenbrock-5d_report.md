# Rosenbrock-5d Optimization Algorithm Benchmark Report

*Generated on 2025-03-10 20:15:11*

## Benchmark Overview

**Test Functions:** rosenbrock

**Algorithms:** cmaes, differential_evolution, dual_annealing, mapio

## Summary Results

| Function | Dimension | cmaes Best Value | differential_evolution Best Value | dual_annealing Best Value | mapio Best Value |
| --- | --- | --- | --- | --- | --- |
| rosenbrock | 5 | 3.6284e-01 | 3.2895e-04 | 2.8420e-11 | 1.9531e-11 |

## Visualization Summary

![Best Value Comparison](rosenbrock-5d_best_value_comparison.png)

![Modes Found Comparison](rosenbrock-5d_modes_found_comparison.png)

## rosenbrock Function

**Description:** Function with a narrow valley leading to the global minimum.

### Convergence Plot

![Convergence Plot](rosenbrock-5d_convergence_rosenbrock.png)

### 2D Exploration

![2D Exploration](rosenbrock-5d_exploration_2d_rosenbrock.png)

### Search Density

![Search Density](rosenbrock-5d_density_rosenbrock_dims.png)

### 3D Exploration

![3D Exploration](rosenbrock-5d_exploration_3d_rosenbrock.png)

### Algorithm Performance

| Algorithm | Best Value | Modes Found |
| --- | --- | --- |
| mapio | 1.953087e-11 | 43 |
| cmaes | 3.628416e-01 | 1 |
| differential_evolution | 3.289514e-04 | 1 |
| dual_annealing | 2.841964e-11 | 1 |

