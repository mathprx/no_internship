from scipy.linalg import solve
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

def interpolate_field (points,means,basis,basis_means=None):

    assert len(points)-1 == len(means), "The number of points must be one more than the number of means"
    assert points[0] == 0, "The first point must be 0"
    assert points[-1] == 1, "The last point must be 1"

    n = len(points)-1

    if basis_means == None :
        basis_means = np.array([[quad(f,points[k],points[k+1])[0]/(points[k+1]-points[k]) for f in basis] for k in range(n)])
        print("basis_means",basis_means)

    coefficients = solve(basis_means,means)

    return coefficients, basis


def vizualize_interpolation(points, means, coefficients, basis):

    x_vals = np.linspace(0, 1, 300)
    y_vals = np.array([sum(coefficients[i] * basis[i](x) for i in range(len(basis))) for x in x_vals])

    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, y_vals, label="Interpolation", color='blue')
    plt.xlabel("x")
    plt.ylabel("f(x)")

    # Visual means
    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        plt.hlines(means[i], a, b, colors='red', linestyles='dashed', label="means" if i == 0 else "")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("interpolation.png")

def test_vizualize_interpolation(points, means, coefficients_list, basis_list, f):
    x_vals = np.linspace(0, 1, 300)
    plt.figure(figsize=(8, 4))
    for coefficients, basis in zip(coefficients_list, basis_list):
        y_vals = np.array([sum(coefficients[i] * basis[i](x) for i in range(len(basis))) for x in x_vals])
        plt.plot(x_vals, y_vals)

    plt.plot(x_vals, f(x_vals), label="True function", color='green', linestyle='dotted')

    for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
        plt.hlines(means[i], a, b, colors='red', linestyles='dashed', label="means" if i == 0 else "")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("interpolation_test.png")



if __name__ == "__main__":
    # Example usage
    points = np.linspace(0, 1, 61) 
    print("points",points)
    polynomials = [(lambda x,i=i : x**i ) for i in range(len(points)-1)]
    cosines = [(lambda x, i=i : np.cos(i * 2 * np.pi * x) ) for i in range(len(points)-1)]
    f = lambda x : np.cos(5*x)
    means = np.array([quad(f, points[k], points[k+1])[0] / (points[k+1] - points[k]) for k in range(len(points)-1)])

    coefficients_list = []
    basis_list = []
    poly_coefficients, _ = interpolate_field(points, means, polynomials)
    coefficients_list.append(poly_coefficients)
    basis_list.append(polynomials)
    cos_coefficients, _ = interpolate_field(points, means, cosines)
    # coefficients_list.append(cos_coefficients)
    # basis_list.append(cosines)

    test_vizualize_interpolation(points, means, coefficients_list, basis_list,f)


