import numpy as np
import matplotlib.pyplot as plt
class ConstrainedOptimizer:
    def __init__(self, f, grad_f, hess_f, constraints):
        self.f = f
        self.grad_f = grad_f
        self.hess_f = hess_f
        self.constraints = constraints
    def _barrier_func(self, x, mu):
        phi = -np.sum([np.log(-g(x)) for g, _, _ in self.constraints])
        return self.f(x) + mu * phi
    def _barrier_grad(self, x, mu):
        g_grads = []
        for g, grad_g, _ in self.constraints:
            val_g = g(x)
            g_grads.append((1.0 / -val_g) * grad_g(x))
        return self.grad_f(x) + mu * np.sum(g_grads, axis=0)
    def _barrier_hess(self, x, mu):
        hess_total = self.hess_f(x)
        for g, grad_g, hess_g in self.constraints:
            val_g = g(x)
            gr = grad_g(x).reshape(-1, 1)
            term = (1.0 / val_g**2) * (gr @ gr.T) - (1.0 / val_g) * hess_g(x)
            hess_total += mu * term
        return hess_total
    def _is_feasible(self, x):
        return all(g(x) < 0 for g, _, _ in self.constraints)
    def solve(self, x0, mu_start=1.0, mu_factor=0.1, tol=1e-6, max_outer=20):
        x = np.array(x0, dtype=float)
        mu = mu_start
        m = len(self.constraints)
        history = [x.copy()]
        print(f"{'Iter':<10} | {'mu':<10} | {'f(x)':<15} | {'Gap (m*mu)':<10}")
        print("-" * 55)
        for i in range(max_outer):
            for _ in range(50):
                grad = self._barrier_grad(x, mu)
                if np.linalg.norm(grad) < 1e-8:
                    break
                hess = self._barrier_hess(x, mu)
                delta_x = np.linalg.solve(hess, -grad)
                alpha = 1.0
                beta = 0.5
                c1 = 1e-4
                while not self._is_feasible(x + alpha * delta_x):
                    alpha *= beta
                phi_current = self._barrier_func(x, mu)
                while self._barrier_func(x + alpha * delta_x, mu) > phi_current + c1 * alpha * grad @ delta_x:
                    alpha *= beta
                x += alpha * delta_x
            history.append(x.copy())
            print(f"{i:<10} | {mu:<10.2e} | {self.f(x):<15.6f} | {m*mu:<10.2e}")
            if m * mu < tol:
                break
            mu *= mu_factor
        return x, np.array(history)
def cylinder_problem():
    V_min = 100
    f = lambda x: 2 * np.pi * x[0]**2 + 2 * np.pi * x[0] * x[1]
    grad_f = lambda x: np.array([4 * np.pi * x[0] + 2 * np.pi * x[1], 2 * np.pi * x[0]])
    hess_f = lambda x: np.array([[4 * np.pi, 2 * np.pi], [2 * np.pi, 0.0]])
    g1 = lambda x: V_min - np.pi * x[0]**2 * x[1]
    grad_g1 = lambda x: np.array([-2 * np.pi * x[0] * x[1], -np.pi * x[0]**2])
    hess_g1 = lambda x: np.array([[-2 * np.pi * x[1], -2 * np.pi * x[0]], [-2 * np.pi * x[0], 0.0]])
    g2 = lambda x: x[0] - x[1]
    grad_g2 = lambda x: np.array([1.0, -1.0])
    hess_g2 = lambda x: np.zeros((2, 2))
    constraints = [(g1, grad_g1, hess_g1), (g2, grad_g2, hess_g2)]
    opt = ConstrainedOptimizer(f, grad_f, hess_f, constraints)
    x0 = np.array([4.0, 5.0])
    sol, hist = opt.solve(x0)
    print(f"\nSolution optimale : r = {sol[0]:.4f}, h = {sol[1]:.4f}")
    print(f"Surface minimale : {f(sol):.4f}")
    r_vals = np.linspace(1, 6, 100)
    h_vals = np.linspace(1, 10, 100)
    R, H = np.meshgrid(r_vals, h_vals)
    Z = 2 * np.pi * R**2 + 2 * np.pi * R * H
    plt.figure(figsize=(10, 6))
    plt.contour(R, H, Z, levels=30, cmap='viridis')
    plt.plot(hist[:, 0], hist[:, 1], 'ro-', label='Chemin Central')
    plt.fill_between(r_vals, V_min/(np.pi*r_vals**2), 10, where=(r_vals <= 10), color='gray', alpha=0.2, label='Contrainte Volume')
    plt.plot(r_vals, r_vals, 'k--', label='r = h')
    plt.title("Convergence par methode de points interieurs")
    plt.xlabel("Rayon (r)")
    plt.ylabel("Hauteur (h)")
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    cylinder_problem()
