# Rosenbrock-10d Optimization Algorithm Benchmark Report

*Generated on 2025-03-10 20:18:15*

## Benchmark Overview

**Test Functions:** rosenbrock

**Algorithms:** cmaes, differential_evolution, dual_annealing, mapio

## Summary Results

| Function | Dimension | cmaes Best Value | differential_evolution Best Value | dual_annealing Best Value | mapio Best Value |
| --- | --- | --- | --- | --- | --- |
| rosenbrock | 10 | 7.3871e+00 | 2.1845e+00 | 6.6571e-10 | 7.8881e-08 |

## Visualization Summary

![Best Value Comparison](rosenbrock-10d_best_value_comparison.png)

![Modes Found Comparison](rosenbrock-10d_modes_found_comparison.png)

## rosenbrock Function

**Description:** Function with a narrow valley leading to the global minimum.

### Convergence Plot

![Convergence Plot](rosenbrock-10d_convergence_rosenbrock.png)

### 2D Exploration

![2D Exploration](rosenbrock-10d_exploration_2d_rosenbrock.png)

### Search Density

![Search Density](rosenbrock-10d_density_rosenbrock_dims.png)

### 3D Exploration

![3D Exploration](rosenbrock-10d_exploration_3d_rosenbrock.png)

### Algorithm Performance

| Algorithm | Best Value | Modes Found |
| --- | --- | --- |
| mapio | 7.888114e-08 | 62 |
| cmaes | 7.387064e+00 | 1 |
| differential_evolution | 2.184456e+00 | 1 |
| dual_annealing | 6.657051e-10 | 1 |

