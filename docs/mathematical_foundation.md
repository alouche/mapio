# Mathematical Foundation

We adopt the following notation:

- ![f : \mathbb{R}^d \rightarrow \mathbb{R}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}f%20:%20\mathbb{R}^d%20\rightarrow%20\mathbb{R}): Objective function  
- ![\mathcal{X} \subset \mathbb{R}^d](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathcal{X}%20\subset%20\mathbb{R}^d): Bounded search space  
- ![M](https://latex.codecogs.com/png.latex?\fg{FFFFFF}M): Number of parallel paths  
- ![K](https://latex.codecogs.com/png.latex?\fg{FFFFFF}K): Steps per path  
- ![\tau(t)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\tau(t)): Temperature schedule  
- ![\sigma(t)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\sigma(t)): Adaptive step size

## 1. Path Integral Formulation

### 1.1 Stochastic Path Dynamics

Each path ![\mathbf{x}_i^{(k)}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathbf{x}_i^{(k)}) evolves via modified Langevin dynamics:

![\mathbf{x}_i^{(k+1)} = \mathbf{x}_i^{(k)} + \underbrace{\alpha \mathbf{v}_i^{(k)}}_{\text{Momentum}} + \underbrace{\beta \epsilon_i^{(k)}}_{\text{Stochastic Drive}}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathbf{x}_i^{(k+1)}%20=%20\mathbf{x}_i^{(k)}%20+%20\underbrace{\alpha%20\mathbf{v}_i^{(k)}}_{\text{Momentum}}%20+%20\underbrace{\beta%20\epsilon_i^{(k)}}_{\text{Stochastic%20Drive}})

![\mathbf{v}_i^{(k+1)} = \beta \mathbf{v}_i^{(k)} + (1 - \beta) \epsilon_i^{(k)}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathbf{v}_i^{(k+1)}%20=%20\beta%20\mathbf{v}_i^{(k)}%20+%20(1%20-%20\beta)%20\epsilon_i^{(k)})

where ![\epsilon_i^{(k)} \sim \mathcal{N}(0, \Sigma_i^{(k)})](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\epsilon_i^{(k)}%20\sim%20\mathcal{N}(0,%20\Sigma_i^{(k)})) with covariance:

![\Sigma_i^{(k)} = \text{diag} \left( \sigma_i^2 \left[ 1 + \kappa \rho_j^{(k)} \right]^{-1} \right)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\Sigma_i^{(k)}%20=%20\text{diag}%20\left(%20\sigma_i^2%20\left[%201%20+%20\kappa%20\rho_j^{(k)}%20\right]^{-1}%20\right))

### 1.2 Path Weight Calculation (Generalized Simpson's Rule)

For path ![i](https://latex.codecogs.com/png.latex?\fg{FFFFFF}i) with values ![\{f_i^{(1)}, \ldots, f_i^{(K)}\}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\{f_i^{(1)},%20\ldots,%20f_i^{(K)}\}):

![w_i = \frac{1}{3} \sum_{m=1}^{K/2} \left[ e^{-f_i^{(2m-2)}/\tau} + 4e^{-f_i^{(2m-1)}/\tau} + e^{-f_i^{(2m)}/\tau} \right] \Delta k](https://latex.codecogs.com/png.latex?\fg{FFFFFF}w_i%20=%20\frac{1}{3}%20\sum_{m=1}^{K/2}%20\left[%20e^{-f_i^{(2m-2)}/\tau}%20+%204e^{-f_i^{(2m-1)}/\tau}%20+%20e^{-f_i^{(2m)}/\tau}%20\right]%20\Delta%20k)

where ![\Delta k = \frac{(k_{max} - k_{min})}{(K - 1)}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\Delta%20k%20=%20\frac{(k_{max}%20-%20k_{min})}{(K%20-%201)}). For odd ![K](https://latex.codecogs.com/png.latex?\fg{FFFFFF}K), the last interval uses trapezoidal rule.


## 2. Thermodynamic Analogies

### 2.1 Adaptive Temperature Schedule

![\tau(t) = \underbrace{\frac{\tau_0}{\sqrt{1 + t/T_0}}}_{\text{Base Cooling}} \cdot \underbrace{\left( 1 + A \sin \left( \frac{2\pi t}{T_1} \right) \right)}_{\text{Exploration Cycles}} \cdot \underbrace{\tanh \left( \frac{D(t)}{D_0} \right)}_{\text{Diversity Control}}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\tau(t)%20=%20\underbrace{\frac{\tau_0}{\sqrt{1%20+%20t/T_0}}}_{\text{Base%20Cooling}}%20\cdot%20\underbrace{\left(%201%20+%20A%20\sin%20\left(%20\frac{2\pi%20t}{T_1}%20\right)%20\right)}_{\text{Exploration%20Cycles}}%20\cdot%20\underbrace{\tanh%20\left(%20\frac{D(t)}{D_0}%20\right)}_{\text{Diversity%Control}})

where diversity ![D(t) = \frac{1}{M} \sum_{i=1}^M \| \mathbf{x}_i - \bar{\mathbf{x}} \|^2](https://latex.codecogs.com/png.latex?\fg{FFFFFF}D(t)%20=%20\frac{1}{M}%20\sum_{i=1}^M%20\\|%20\\mathbf{x}_i%20-%20\\bar{\mathbf{x}}\\|^2) and ![D_0 = \text{Var}(\mathcal{X})](https://latex.codecogs.com/png.latex?\fg{FFFFFF}D_0=%5Ctext{%7BVar%7D}(\\mathcal{%7BX}).)

### 2.2 Free Energy Minimization

The algorithm implicitly minimizes:

![\mathcal{F} = \mathbb{E}[f] - \tau \mathcal{H}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathcal{F}%20=%20\mathbb{E}[f]%20-%20\tau%20\mathcal{H})

where ![\mathcal{H} = -\sum_{j=1}^{N_{\text{modes}}} p_j \ln p_j](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathcal{H}%20=%20-\sum_{j=1}^{N_{\text{modes}}}%20p_j%20\ln%20p_j) is the entropy of mode probabilities.

## 3. Mode Management System

### 3.1 Voronoi Allocation

For ![N_{modes}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}N_{modes}) discovered modes ![\{ \mathbf{m}_j \}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\{%20\mathbf{m}_j%20\}):

1. Estimate Voronoi cell volumes via Monte Carlo:

![V_j \approx \frac{1}{N_{samples}} \sum_{n=1}^{N_{samples}} \mathbb{I}(\|\mathbf{x}_n - \mathbf{m}_j\| < \|\mathbf{x}_n - \mathbf{m}_k\| \ \forall k \neq j)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}V_j%20\approx%20\frac{1}{N_{samples}}%20\sum_{n=1}^{N_{samples}}%20\mathbb{I}(\|\mathbf{x}_n%20-%20\mathbf{m}_j\|%20<%20\|\mathbf{x}_n%20-%20\mathbf{m}_k\|%20\%20forall%20k%20\neq%20j))

2. Allocation probabilities:

![P_j = \frac{\exp(-f_j / \tau)}{\sum_k \exp(-f_k / \tau)} \cdot \frac{V_j^{2/3}}{\sum_k V_k^{2/3}}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}P_j%20=%20\frac{\exp(-f_j%20/%20\tau)}{\sum_k%20\exp(-f_k%20/%20\tau)}%20\cdot%20\frac{V_j^{2/3}}{\sum_k%20V_k^{2/3}})

### 3.2 Mode Density Gradient

Perturbation strength for path ![i](https://latex.codecogs.com/png.latex?\fg{FFFFFF}i) near mode ![j](https://latex.codecogs.com/png.latex?\fg{FFFFFF}j):

![\gamma_{ij} = \gamma_0 \left( 1 + \frac{\nabla \rho_j \cdot (\mathbf{x}_i - \mathbf{m}_j)}{\|\nabla \rho_j\| \|\mathbf{x}_i - \mathbf{m}_j\|} \right)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\gamma_{ij}%20=%20\gamma_0%20\left(%201%20+%20\frac{\nabla%20\rho_j%20\cdot%20(\mathbf{x}_i%20-%20\mathbf{m}_j)}{\|\nabla%20\rho_j\|%20\|\mathbf{x}_i%20-%20\mathbf{m}_j\|}%20\right))

where

![\rho_j = \frac{1}{\pi \tau} \sum_{k=1}^{M} \exp \left( -\frac{\|\mathbf{x}_i - \mathbf{m}_k\|^2}{2 \tau} \right)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\rho_j%20=%20\frac{1}{\pi%20\tau}%20\sum_{k=1}^{M}%20\exp%20\left(%20-\frac{\|\mathbf{x}_i%20-%20\mathbf{m}_k\|^2}{2%20\tau}%20\right))

## 4. Landscape-Adaptive Parameter Control

### 4.1 Ruggedness Metric

![R = \frac{\sigma_f}{\Delta f} = \frac{1}{N_{samples}} \sum_{i<j} \left| \frac{f(\mathbf{x}_i) - f(\mathbf{x}_j)}{\|\mathbf{x}_i - \mathbf{x}_j\|} \right|](https://latex.codecogs.com/png.latex?\fg{FFFFFF}R%20=%20\frac{\sigma_f}{\Delta%20f}%20=%20\frac{1}{N_{samples}}%20\sum_{i<j}%20\left|%20\frac{f(\mathbf{x}_i)%20-%20f(\mathbf{x}_j)}{\|\mathbf{x}_i%20-%20\mathbf{x}_j\|}%20\right|)

### 4.2 Multimodality Index

![\mathcal{M} = \frac{N_{\text{modes}}}{N_{\text{samples}}} + \frac{1}{\langle \| \mathbf{m}_i - \mathbf{m}_k \| \rangle}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathcal{M}%20=%20\frac{N_{\text{modes}}}{N_{\text{samples}}}%20+%20\frac{1}{\langle%20\|%20\mathbf{m}_i%20-%20\mathbf{m}_k%20\|%20\rangle})


### 4.3 Parameter Update Laws

| Parameter | Update Rule                                                                                                                      |
|-----------|----------------------------------------------------------------------------------------------------------------------------------|
| σ         | ![\sigma \leftarrow \sigma_0 \cdot \tanh(R / \mathcal{M})](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\sigma%20\leftarrow%20\sigma_0%20\cdot%20\tanh(R%20/%20\mathcal{M})) |
| λ         | ![\lambda \leftarrow \lambda_0 \cdot (1 - e^{-\mathcal{M}t})](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\lambda%20\leftarrow%20\lambda_0%20\cdot%20(1%20-%20e^{-\mathcal{M}t})) |
| τ₀        | ![\tau_0 \leftarrow \tau_0 \cdot (1 + 0.5 \sin(2\pi t / T_{reheat}))](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\tau_0%20\leftarrow%20\tau_0%20\cdot%20(1%20+%200.5%20\sin(2\pi%20t%20/%20T_{reheat}))) |


## 5. Convergence Analysis

### 5.1 Lyapunov Function Candidate

![\mathcal{V}(t) = \underbrace{\frac{1}{M} \sum_{i=1}^{M} f(x_i)}_{\text{Performance}} + \underbrace{\tau(t) \log N_{modes}}_{\text{Entropic Cost}} + \underbrace{\frac{\eta}{2} \|\sigma(t) - \sigma^*\|^2}_{\text{Parameter Adaptation}}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\mathcal{V}(t)%20=%20\underbrace{\frac{1}{M}%20\sum_{i=1}^{M}%20f(x_i)}_{\text{Performance}}%20+%20\underbrace{\tau(t)%20\log%20N_{modes}}_{\text{Entropic%20Cost}}%20+%20\underbrace{\frac{\eta}{2}%20\|\sigma(t)%20-%20\sigma^*\|^2}_{\text{Parameter%20Adaptation}})

### 5.2 First-Order Dynamics

![\frac{d\mathcal{V}}{dt} \leq -\alpha \sum_{i=1}^{M} \|\nabla f(x_i)\|^2 + \beta \tau(t) \frac{dN_{modes}}{dt} + \mathcal{O}(\sigma^2)](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\frac{d\mathcal{V}}{dt}%20\leq%20-\alpha%20\sum_{i=1}^{M}%20\|\nabla%20f(x_i)\|^2%20+%20\beta%20\tau(t)%20\frac{dN_{modes}}{dt}%20+%20\mathcal{O}(\sigma^2))

Guarantees convergence to ![\epsilon](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\epsilon)-neighborhood of critical points when:

![\alpha > \frac{\beta \tau_0}{\gamma \min_j V_j}](https://latex.codecogs.com/png.latex?\fg{FFFFFF}\alpha%20>%20\frac{\beta%20\tau_0}{\gamma%20\min_j%20V_j})