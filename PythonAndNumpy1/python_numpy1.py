# python_intro.py
"""Python Essentials: Introduction to Python.
Sarah Beethe
MTH 520
4/15/2022
"""

if __name__ == "__main__":
    pass                    #pass is a temporary placeholder

#%%
#Problem 2
# a classic hello world intro script 

print("Hello, World") #printing a string to the terminal window

#%%
#Problem 3
# A function that returns the volume of a sphere accepting a parameter r
def sphere_volume(r):
    pi = 3.14159  
    v = 4/3*pi* r**3
    return v
# Note: To change the value of r, you must change the r in the print line
print(sphere_volume(6))

#%%
#Problem 4
#A function that defines two matrices as NumPy arrays and returns the matrix product AB

import numpy as np

def prob4():
    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    prod = np.dot(A, B)
    print('Matrix A is: \n', A, '\nMatrix B is: \n', B)
    return prod

print('The product of matrix A and B is: \n', prob4())

#%%

# Problem 5
