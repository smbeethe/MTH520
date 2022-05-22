# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Sarah Beethe>
<MTH 520>
<05/19/22>
"""


# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la

# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    q, r = la.qr(A, mode = 'economic')
    y = np.dot(q.T, b)
    x = la.solve(r, y)
    return x
    
A = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
b = np.array([3, 2, 1])
print(least_squares(A, b))


  
# Problem 2


def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    
    a, b = np.load(r"\\wsl.localhost\Ubuntu\home\beethes\PythonEssentials\LeastSquares_Eigenvalues\housing.npy").T
    A = np.vstack((a, np.ones_like(a))).T
    B = b.T
    q, r = (least_squares(A, B))
    print(A, B)
    plt.scatter(A[:, 0], B)
    
    x_ax = np.linspace(0, 16, 100) 
    L = q*x_ax + r
    plt.plot(x_ax, L)
    
    plt.show()
    
line_fit()   


#%%
# Problem 3

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    a, b = np.load(r"\\wsl.localhost\Ubuntu\home\beethes\PythonEssentials\LeastSquares_Eigenvalues\housing.npy").T
    A = np.vstack((a, np.ones_like(a))).T
    B = b.T
    plt.scatter(A[:, 0], B)
    
    3deg = polyld(polyfit.)
    
    x_ax = np.linspace(0, 16, 100) 
    lsqa, lsqb = la.lstsq(A, B)[0]
    lin = lsqa*x_ax + lsqb
    plt.plot(x_ax, lin)
    fit = np.poly1d(A[:,0])
    print(lsqa, lsqb, fit)
    plt.show()

polynomial_fit()
    



def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")


