import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from random import random

# === Data ===
segments_test = [(0, 1), (1, 2), (2, 3)]
means_test = [2.0, 1.0, 3.0]



class soft_interpolation:
    def __init__(self, segments=segments_test, means=means_test):
        self.n = len(segments)
        # === Verification of proper segment connections ===
        assert all(np.isclose(segments[i][1], segments[i+1][0]) for i in range(self.n - 1)), "The segments are not properly connected."

        self.segments = segments
        self.means = means
        self.solved = False
        self.coeffs = None
    
    # === Function f_i(x) = a + b*(x - x0) + c*(x - x0)^2 ===
    def f_piecewise(self, x, coeffs):
        for i, (x0, x1) in enumerate(self.segments):
            if x0 <= x <= x1:
                a = coeffs[3*i]
                b = coeffs[3*i + 1]
                c = coeffs[3*i + 2]
                return a + b*(x - x0) + c*(x - x0)**2
        return 0

    # === Functional to minimize: ∑ ∫(f')² dx on each segment
    def objective(self, coeffs):
        energy = 0
        for i, (x0, x1) in enumerate(self.segments):
            h = x1 - x0
            b = coeffs[3*i + 1]
            c = coeffs[3*i + 2]
            energy += b**2 * h + 2 * b * c * h**2 + (4/3) * c**2 * h**3
        return energy

    # === Constraints: mean value of f on each segment
    def make_mean_constraint(self, i):
        x0, x1 = self.segments[i]
        h = x1 - x0
        mi = self.means[i]
        def constraint(coeffs):
            a = coeffs[3*i]
            b = coeffs[3*i + 1]
            c = coeffs[3*i + 2]
            integral = a*h + 0.5*b*h**2 + (1/3)*c*h**3
            return integral / h - mi
        return constraint

    # === Constraints: continuity of f
    def make_continuity_constraint(self, i):
        x0, x1 = self.segments[i]
        def constraint(coeffs):
            a1, b1, c1 = coeffs[3*i:3*i+3]
            a2 = coeffs[3*(i+1)]
            return a1 + b1*(x1 - x0) + c1*(x1 - x0)**2 - a2
        return constraint

    # === Constraints: continuity of f'
    def make_derivative_constraint(self, i):
        x0, x1 = self.segments[i]
        def constraint(coeffs):
            b1, c1 = coeffs[3*i + 1], coeffs[3*i + 2]
            b2 = coeffs[3*(i+1) + 1]
            return b1 + 2*c1*(x1 - x0) - b2
        return constraint

    def solve(self):
        # === Building constraints
        constraints = [{'type': 'eq', 'fun': self.make_mean_constraint(i)} for i in range(self.n)]
        constraints += [{'type': 'eq', 'fun': self.make_continuity_constraint(i)} for i in range(self.n - 1)]
        constraints += [{'type': 'eq', 'fun': self.make_derivative_constraint(i)} for i in range(self.n - 1)]

        # === Initial point: piecewise constant function
        x0 = np.zeros(3 * self.n)
        for i in range(self.n):
            x0[3*i] = self.means[i]  # a_i = mean
            x0[3*i + 1] = 1e-5 *  random()         #  almost zero slope
            x0[3*i + 2] = 1e-5 *  random()         # almost zero curvature

        # === Optimization
        result = minimize(
            self.objective,
            x0,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 10000, 'disp': True}
        )

        # === Result
        if result.success:
            self.coeffs = result.x
            self.solved = True
            print("✅ Optimization successful.")
            for i in range(self.n):
                a, b, c = self.coeffs[3*i:3*i+3]
                print(f"Segment {i+1}: f(x) ≈ {a:.3f} + {b:.3f}*(x - {self.segments[i][0]}) + {c:.3f}*(x - {self.segments[i][0]})²")
        else:
            self.solved = False
            print("❌ Optimization failed.")
            exit()

    def visualize(self):
        # === Visualization
        x_vals = np.linspace(self.segments[0][0], self.segments[-1][1], 300)
        y_vals = np.array([self.f_piecewise(x, self.coeffs) for x in x_vals])

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, label="Smooth quadratic approximation", color='blue')
        plt.title("Piecewise quadratic approximation (C¹)")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        # Visual means
        for i, (a, b) in enumerate(self.segments):
            plt.hlines(self.means[i], a, b, colors='red', linestyles='dashed', label="Means" if i == 0 else "")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # === Test ===
    interpolator = soft_interpolation(segments=segments_test, means=means_test)
    interpolator.solve()
    interpolator.visualize()