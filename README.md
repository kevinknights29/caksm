# caksm — Communication-Avoiding Krylov Subspace Methods for Option Pricing

PDE-based pricer for two 3-asset European options (basket call and rainbow min-call), written in C++23. 
Five numerical methods are implemented: Crank-Nicolson, two ADI variants, Matrix Exponential, and a Krylov subspace exponential integrator.

## Dependencies

| Library  | Source                           | Purpose                                         |
|----------|----------------------------------|-------------------------------------------------|
| `Eigen`  | fetched via CMake `FetchContent` | Sparse linear algebra, dense matrix exponential |
| `Catch2` | fetched via CMake `FetchContent` | Unit testing framework                          |

No system-level installs are required. CMake downloads Eigen and Catch2 automatically on first configure.

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

The binary is `build/pricer`. An optional compile-commands symlink improves IDE integration:

```bash
ln -sf build/compile_commands.json compile_commands.json
```

### Run tests

```bash
cd build && ctest --output-on-failure
```

## Usage

```
./build/pricer [OPTIONS]
```

| Option                     | Default             | Description                                |
|----------------------------|---------------------|--------------------------------------------|
| `--benchmark`              | *(required to run)* | Run all five methods for both option types |
| `--option basket\|rainbow` | `basket`            | Select which option type to price          |
| `--n N`                    | `15`                | Grid points per spatial dimension          |
| `--steps M`                | `100`               | Number of temporal steps                   |
| `--help`                   |                     | Print usage summary and exit               |

### Examples

```bash
# Default benchmark: both option types, n=15, 100 steps
./build/pricer --benchmark

# Rainbow only, finer grid
./build/pricer --benchmark --option rainbow --n 20 --steps 200

# Basket only, coarse grid for quick sanity check
./build/pricer --benchmark --option basket --n 10 --steps 50
```

### Sample output

```
European Option PDE Pricer [Benchmark]

 Basket call
  Parameters: n=15, steps=100, K=100, r=0.04, T=1.0
  sigma=[0.30,0.35,0.40]  rho_off=[0.5,0.5,0.5]
  Reference price: 13.2449

  Method            Price     Error    Time(ms)
  ----------------------------------------------
  [Building PDE system...]
  CN              13.1792    0.0657        90.1
  ADI-DR          13.1866    0.0583        19.1
  ADI-HV          13.1792    0.0657        34.6
  ME              13.1792    0.0657        18.2
  KSM-EI          13.1792    0.0657        25.2


 Rainbow min-call
  Parameters: n=15, steps=100, K=100, r=0.04, T=1.0
  sigma=[0.30,0.35,0.40]  rho_off=[0.5,0.5,0.5]
  Reference price: 4.4450

  Method            Price     Error    Time(ms)
  ----------------------------------------------
  [Building PDE system...]
  CN               4.0851    0.3599        85.2
  ADI-DR           4.0918    0.3532        17.5
  ADI-HV           4.0852    0.3598        33.1
  ME               4.0851    0.3599        15.4
  KSM-EI           4.0851    0.3599        24.9
```

Grid error for the rainbow option at n=15 is expected; increase `--n` for higher accuracy.

## Default model parameters

| Parameter                | Value              | Description                                                 |
|--------------------------|--------------------|-------------------------------------------------------------|
| Spot prices $S_0$        | (100, 100, 100)    | Initial asset prices                                        |
| Strike K                 | 100                | Option strike                                               |
| Risk-free rate r         | 0.04               | Continuously compounded                                     |
| Maturity T               | 1.0 year           | Fixed                                                       |
| Volatilities $\sigma$    | (0.30, 0.35, 0.40) | Per-asset annual vol                                        |
| Correlations $\rho$      | (0.50, 0.50, 0.50) | Off-diagonal pairs ($\rho_{01}, \rho_{02}, \rho_{03}$)      |
| Basket weights w         | (1/3, 1/3, 1/3)    | Equal-weight basket                                         |
| Grid half-width $\alpha$ | 2.85               | Log-price domain $\pm \alpha \sigma \sqrt{T}$ per dimension |


## Numerical methods

| Method   | Scheme                                                | Notes                                       |
|----------|-------------------------------------------------------|---------------------------------------------|
| `CN`     | Crank-Nicolson ($\theta = 0.5$)                       | Fully implicit 2nd-order; SparseLU per run  |
| `ADI-DR` | Douglas-Rachford ADI ($\theta = 0.5$)                 | 3 direction-split solves per step           |
| `ADI-HV` | Hundsdorfer-Verwer ADI ($\theta = 0.5, \sigma = 0.5$) | DR predictor + HV corrector                 |
| `ME`     | Matrix Exponential (scaling-and-squaring)             | Taylor polynomial, $\infty$-norm early exit |
| `KSM-EI` | Krylov subspace exponential integrator                | Incremental Arnoldi; expm on small H_m      |

## Model

3-asset Black-Scholes PDE in log-price coordinates, solved backward in pseudo-time $\tau$ from 0 (payoff) to T (price):

$$
\frac{\partial u}{\partial \tau} = \frac{1}{2} \sum_{d,d'} \rho_{dd'} \sigma_d \sigma_{d'} \frac{\partial^2 u}{\partial x_d \partial x_{d'}} + \left(r - \tfrac{1}{2}\sigma_d^2\right) \frac{\partial u}{\partial x_d} - r \, u + b(\tau)
$$

**Basket call payoff:**

$$
u_0(x) = \max\!\left(\sum_d w_d e^{x_d} - K,\ 0\right)
$$

**Rainbow min-call payoff:**

$$
u_0(x) = \max\!\left(\min_d e^{x_d} - K,\ 0\right)
$$

The boundary forcing term $b(\tau)$ for the basket case is encoded as a polynomial $B \cdot s(\tau)$ with $s(\tau) = [\tau^2/2,\, \tau,\, 1]^\top$, derived from a deep-ITM approximation on ghost nodes outside the grid. 
The rainbow case uses modified finite-difference stencils (zero-gamma boundary condition) instead.
