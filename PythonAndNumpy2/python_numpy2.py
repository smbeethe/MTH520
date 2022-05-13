# python_intro.py
"""Python Essentials: Introduction to Python.
Sarah Beethe
MTH 520
4/15/2022
"""
#%%
#Problem 1
def isolate(a, b, c, d, e):
    print(a, "   ", b, "   ", c, d, e)

isolate(1, 2, 3, 4, 5)

#%%
#Problem 2
def first_half(string):
    half_string = len(string)//2
    return string[:half_string]
    raise NotImplementedError("Problem 2 Incomplete")
print(first_half("I am having fun doing my homework"))

def backward(first_string):
    return first_string[slice(None, None, -1)]
    raise NotImplementedError("Problem 2 Incomplete")
print(backward("I am having fun doing my homework"))

#%%
#Problem 3
def list_ops():
    list = ["bear", "ant", "cat", "dog"]
    list.append("eagle")
    list[2] = "fox"
    list.pop(1)
    list.sort(reverse = True)
    list[list.index("eagle")] = "hawk"
    list[3] = list[3] + "hunter"
    return list
    raise NotImplementedError("Problem 3 Incomplete")
print(list_ops())

#%%
#Problem 4
import numpy as np

def alt_harmonic(n):
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    N = sum([(-1)**(i+1)/float(i) for i in range (1, n+1)])
    return N
    
    raise NotImplementedError("Problem 4 Incomplete")
print(alt_harmonic(500000))




#%%

import numpy as np

def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = np.array(A)
    mask = B < 0 
    B[mask] = 0
    
    return B
    
    raise NotImplementedError("Problem 5 Incomplete")
        
print(prob5([-1, -3, 3]))
    
    

#%%
def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
    AT = np.transpose(A)
    I = np.identity(3)
    Z1 = np.zeros((3,3))
    Z2 = np.zeros((2,2))
    Z32 = np.zeros((3, 2))
    Z23 = np.zeros((2, 3))
    col1 = np.vstack((Z1, A, B))
    col2 = np.vstack((AT, Z2, Z32))
    col3 = np.vstack((I, Z23, C))
    block = np.hstack((col1, col2, col3))
    return block

print(prob6())
    
   # raise NotImplementedError("Problem 6 Incomplete")

#%%
#DO NOT DO
#def prob7(A):
#    """Divide each row of 'A' by the row sum and return the resulting array.

#    Example:
#        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
#        >>> prob6(A)
#        array([[ 0.5       ,  0.5       ,  0.        ],
#               [ 0.        ,  1.        ,  0.        ],
#               [ 0.33333333,  0.33333333,  0.33333333]])
#    """
#    raise NotImplementedError("Problem 7 Incomplete")
#%%
#DONOTDO

#def prob8():
 #   """Given the array stored in grid.npy, return the greatest product of four
  #  adjacent numbers in the same direction (up, down, left, right, or
   # diagonally) in the grid.
    #"""
 #   raise NotImplementedError("Problem 8 Incomplete")
    

    


