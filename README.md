# Numerical analysis & computational mathematics
**High performance implementations developed during my L2 of Mathematics.**

This repository illustrates my ability to link theoretical mathematical concepts to efficient programming. These projects were developed alongside my university studies, reflecting a strong focus in numerical stability and algorithmic complexity.

---

## Key projects

### 1. Numerical Linear Algebra (C language)
Implementation of the fundamental operations of linear algebra, with an emphasis on stability:
* **QR decomposition via Householder reflections**: Unlike the classical Gram-Schmidt method, this implementation uses Householder reflections to transform a matrix into a product of an orthogonal matrix $Q$ and an upper triangular matrix $R$, ensuring superior numerical stability.
* **LU decomposition**: A robust solver for $Ax = b$ systems, utilizing partial pivoting to handle singular or ill-conditioned matrices.
* **Conjugate gradient & CSR storage**: Implementation of a **krylov subspace method** for large-scale linear systems. It uses **Compressed sparse row (CSR)** storage to optimize memory footprint and CPU cache usage during matrix-vector multiplications.

### 2. High precision ODE solvers (C language)
Applications in dynamical systems and computational physics.
* **RKF45 adaptive integrator**: Development of a **Runge kutta fehlberg 4-5** solver. This implementation features an adaptive step size to maintain precise tolerance levels, essential for simulating stiff differential equations.
* **Lorenz attractor simulation**: Applied the RKF45 solver to model and visualize the chaotic behavior of the Lorenz system, demonstrating high precision numerical integration.

### 3. Non linear optimization (Python)
* **Interior point method**: An optimization algorithm using a **logarithmic barrier function** and backtracking line search to track the central path toward the optimal solution under inequality constraints.

---

## 🛠 Skills & Tools
* **Languages:** C (primary for performance), Python (data science and prototyping).
* **Environment:** Linux, Git, LaTeX (for all scientific reporting).
* **Theoretical Interests:** Topology applied to 3D modeling, Chaotic systems, and numerical linear algebra.

---
## 🛠 Skills & Tools
* **Languages:** C (primary for performance), Python (prototyping and visualization).
* **Environment:** Linux, Git, LaTeX (for all scientific reporting).
