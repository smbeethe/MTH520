# drazin.py
"""Volume 1: The Drazin Inverse.
<Sarah Beethe>
<MTH520>
<5/22/2022>
"""

import numpy as np
from scipy import linalg as la


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    if k == index(A, tol=1e-5):
        AAd = np.dot(A, Ad)
        AdA = np.dot(Ad, A)
        Ak1Ad = np.dot(np.linalg.matrix_power(A, k+1), Ad)
        Ak = np.linalg.matrix_power(A, k)
        AdAAd = np.linalg.multi_dot([Ad, A, Ad])
        cdn_1 = np.allclose(AAd, AdA)
        cdn_2 = np.allclose(Ak1Ad, Ak)
        cdn_3 = np.allclose(AdAAd, Ad)
        return cdn_1, cdn_2, cdn_3
    else:
        return False


A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
Ad = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])
k = 1

#A = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
#Ad = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
#k = 3


print(is_drazin(A, Ad, k))
#%%
# Problem 2


def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    
    
    f = lambda x: abs(x) > tol
    T, Q, k = la.schur(A, sort=f)
    g = lambda x: abs(x) <= tol
    T1, Q1, k1 = la.schur(A, sort=g)
    U = np.hstack((Q[:, :k], Q1[:, :len(Q) - k]))
    U_in = la.inv(U)
    V = np.dot(U_in, np.dot(A, U))
    Z = np.zeros_like(A, dtype=float)
    if k != 0:
        M_in = la.inv(V[:k, :k]) 
        Z[:k, :k] = M_in
    Ad = np.dot(U, np.dot(Z, U_in))
    #check = is_drazin(A, Ad, k1)
    return Ad

#A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
A = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
print(drazin_inverse(A, tol=1e-4))  

#%%  
# Problem 3
def laplacian(A):
    D = A.sum(axis=1)
    return np.diag(D) - A

def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = len(A)
    L = laplacian(A)
    ER = np.zeros_like(A, dtype = float)
    I = np.eye(n)
    
    for j in range(n):
        L_t = np.copy(L)
        L_t[j, :] = I[j, :]
        D = drazin_inverse(L_t)
        ER[:, j] = np.diag(D)
    
    return ER - I
    
A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
print(effective_resistance(A))     





