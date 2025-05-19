import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# === Données ===
segments_test = [(0, 1), (1, 2), (2, 3)]
moyennes_test = [2.0, 1.0, 3.0]



class soft_interpolation:
    def __init__(self, segments=segments_test, moyennes=moyennes_test):
        self.n = len(segments)
        # === Vérification de la bonne jonction des segments ===
        assert all(np.isclose(segments[i][1], segments[i+1][0]) for i in range(self.n - 1)), "Les segments ne sont pas bien joints."

        self.segments = segments
        self.moyennes = moyennes
        self. solved = False
        self.coeffs = None
    
    # === Fonction f_i(x) = a + b*(x - x0) + c*(x - x0)^2 ===
    def f_piecewise(self, x, coeffs):
        for i, (x0, x1) in enumerate(self.segments):
            if x0 <= x <= x1:
                a = coeffs[3*i]
                b = coeffs[3*i + 1]
                c = coeffs[3*i + 2]
                return a + b*(x - x0) + c*(x - x0)**2
        return 0

    # === Fonctionnelle à minimiser : ∑ ∫(f')² dx sur chaque segment
    def objective(self, coeffs):
        energy = 0
        for i, (x0, x1) in enumerate(self.segments):
            h = x1 - x0
            b = coeffs[3*i + 1]
            c = coeffs[3*i + 2]
            energy += b**2 * h + 2 * b * c * h**2 + (4/3) * c**2 * h**3
        return energy

    # === Contraintes : moyenne de f sur chaque segment
    def make_mean_constraint(self, i):
        x0, x1 = self.segments[i]
        h = x1 - x0
        mi = self.moyennes[i]
        def constraint(coeffs):
            a = coeffs[3*i]
            b = coeffs[3*i + 1]
            c = coeffs[3*i + 2]
            integral = a*h + 0.5*b*h**2 + (1/3)*c*h**3
            return integral / h - mi
        return constraint

    # === Contraintes : continuité de f
    def make_continuity_constraint(self, i):
        x0, x1 = self.segments[i]
        def constraint(coeffs):
            a1, b1, c1 = coeffs[3*i:3*i+3]
            a2 = coeffs[3*(i+1)]
            return a1 + b1*(x1 - x0) + c1*(x1 - x0)**2 - a2
        return constraint

    # === Contraintes : continuité de f'
    def make_derivative_constraint(self, i):
        x0, x1 = self.segments[i]
        def constraint(coeffs):
            b1, c1 = coeffs[3*i + 1], coeffs[3*i + 2]
            b2 = coeffs[3*(i+1) + 1]
            return b1 + 2*c1*(x1 - x0) - b2
        return constraint

    def solve(self):
        # === Construction des contraintes
        constraints = [{'type': 'eq', 'fun': self.make_mean_constraint(i)} for i in range(self.n)]
        constraints += [{'type': 'eq', 'fun': self.make_continuity_constraint(i)} for i in range(self.n - 1)]
        constraints += [{'type': 'eq', 'fun': self.make_derivative_constraint(i)} for i in range(self.n - 1)]

        # === Point initial : fonction constante par morceau
        x0 = np.zeros(3 * self.n)
        for i in range(self.n):
            x0[3*i] = self.moyennes[i]  # a_i = moyenne
            x0[3*i + 1] = 0        # pente nulle
            x0[3*i + 2] = 0        # courbure nulle

        # === Optimisation
        result = minimize(
            self.objective,
            x0,
            constraints=constraints,
            method='SLSQP',
            options={'ftol': 1e-9, 'maxiter': 10000, 'disp': True}
        )

        # === Résultat
        if result.success:
            self.coeffs = result.x
            self.solved = True
            print("✅ Optimisation réussie.")
            for i in range(self.n):
                a, b, c = self.coeffs[3*i:3*i+3]
                print(f"Segment {i+1}: f(x) ≈ {a:.3f} + {b:.3f}*(x - {self.segments[i][0]}) + {c:.3f}*(x - {self.segments[i][0]})²")
        else:
            self.solved = False
            print("❌ Optimisation échouée.")
            exit()

    def visualize(self):
        # === Visualisation
        x_vals = np.linspace(self.segments[0][0], self.segments[-1][1], 300)
        y_vals = np.array([self.f_piecewise(x, self.coeffs) for x in x_vals])

        plt.figure(figsize=(8, 4))
        plt.plot(x_vals, y_vals, label="Approximation quadratique douce", color='blue')
        plt.title("Approximation par morceaux quadratiques (C¹)")
        plt.xlabel("x")
        plt.ylabel("f(x)")

        # Moyennes visuelles
        for i, (a, b) in enumerate(self.segments):
            plt.hlines(self.moyennes[i], a, b, colors='red', linestyles='dashed', label="Moyennes" if i == 0 else "")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    prob = soft_interpolation(segments=segments_test, moyennes=moyennes_test)
    prob.solve()
    prob.visualize()