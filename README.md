# fixed-step-odes

A small reference library implementing a standard set of **fixed step-size ODE integrators** across major method families.

This project is intentionally simple and explicit: methods are implemented directly from their defining order conditions and tableau formulations, without adaptivity, error control, or performance-oriented optimizations.

The goal is correctness, clarity, and completeness within a narrow scope.

---

## Features

The library includes the following families of time integrators:

### Explicit Runge–Kutta
- Classical RK methods (RK1–RK4)
- Higher-order explicit RK schemes
- Intended for nonstiff problems or as predictors / bootstrappers

### Implicit Runge–Kutta (Collocation)
- Gauss–Legendre methods
- Radau IIA methods
- Lobatto IIIC methods
- Arbitrary stage count (practical limits apply)

All IRK methods are constructed from generated collocation tableaus and solved via Newton iteration.

### Singly Diagonally Implicit Runge–Kutta (SDIRK)
- Selected standard SDIRK schemes (orders 2–4)
- Fixed step-size, fully implicit per stage

### Linear Multistep Methods
- Adams–Bashforth (explicit)
- Adams–Moulton (implicit)
- Backward Differentiation Formulas (BDF)

Coefficients are generated from exact moment/order conditions rather than hard-coded tables.

---

## Design Philosophy

- **Fixed step size only**  
  No adaptivity or embedded error control.

- **Method transparency**  
  Tableaus and coefficients are generated explicitly from theory.

- **Reference over performance**  
  No sparse solvers, no Jacobian reuse, no preconditioning.

- **Intended as a building block**  
  Suitable as a backend for research codes, SDE solvers, or solver comparisons.

This is not a production integrator and does not aim to compete with libraries such as CVODE, Sundials, or DifferentialEquations.jl.

---

## Example Usage

```python
from rk import solve_rk4
from irk import solve_collocation

def f(t, y):
    return -y

t, y = solve_rk4(f, (0.0, 5.0), y0=[1.0], h=0.01)

t, y = solve_collocation(
    f, (0.0, 5.0), y0=[1.0], h=0.1,
    family="radau", s=3
)
```
---

## Notes on Stability and Order
Because this library exposes full method families, some combinations are mathematically valid but numerically impractical.

For example:

- Very high-order Adams–Bashforth methods are unstable for stiff problems

- High-order BDF methods lose zero stability beyond order 6

- Large-stage implicit Runge–Kutta methods are computationally expensive

These behaviors are expected and reflect classical numerical analysis results.

---

## Intended Audience
- Students learning numerical ODE methods

- Researchers benchmarking solver behavior

- Developers building higher-level solvers (e.g. SDE integrators)

- Anyone wanting a clean, explicit implementation of classical methods
