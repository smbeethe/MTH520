# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
Sarah Beethe
MTH520
05/27/22
"""
#%%
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)
    
    A = np.array([1, 2, 0])
    G = np.array([0, 1, -4])
    P = np.array([2, 10, 3])
    S = np.eye(3)
    constraints = [A @ x <= 3, G @ x <= 1, P @ x >= 12, S @ x >= 0]
    
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    opt_x = x.value
    print(opt_x)
    print(opt_val)
    
    return opt_x, opt_val
prob1()
    

#%%
# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """

    
    x = cp.Variable(len(A.T), nonneg = True)
    objective = cp.Minimize(cp.norm(x, 1))
    
    constraints = [A @ x == b]
    
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    opt_x = x.value
    print(opt_x)
    print(opt_val)
    
    return opt_x, opt_val

A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
b = np.array([7, 4]).T
l1Min(A, b)
    
#%%
# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    
    A = np.array([[1, 1, 0, 0, 0, 0], 
                  [0, 0, 1, 1, 0, 0], 
                  [0, 0, 0, 0, 1, 1], 
                  [-1, 0, -1, 0, -1, 0],
                  [0, -1, 0, -1, 0, -1]])
    b = np.array([7, 2, 4, -5, -8]).T
    
    x = cp.Variable(len(A.T), nonneg = True)
    objective = cp.Minimize(cp.norm(x, 1))
    
    constraints = [A @ x == b]
    
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    opt_x = x.value
    print(opt_x)
    print(opt_val)
    
    return opt_x, opt_val

prob3()

#%%
# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    Q = np.array([[3, 2, 1],
                  [2, 4, 2],
                  [1, 2, 3]])
    r = np.array([3, 0, 1])
    x = cp.Variable(3)
   
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    opt_val = problem.solve()
    opt_x = x.value
    print(opt_x)
    print(opt_val)
    
    return opt_x, opt_val

prob4()
#%%
# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    x = cp.Variable(len(A.T), nonneg = True)
    objective = cp.Minimize(cp.norm(A @ x - b, 2))
    
    constraints = [cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    opt_val = problem.solve()
    opt_x = x.value
    print(opt_x)
    print(opt_val)
    
    return opt_x, opt_val

A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
b = np.array([7, 4]).T

prob5(A, b)

