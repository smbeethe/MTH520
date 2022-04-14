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
"""" a classic hello world intro script """

print("Hello, World") #printing a string to the terminal window

#%%
#Problem 3
"""" A function that returns the volume of a sphere accepting a parameter r"""

def sphere_volume(r):
    pi = 3.14159  
    v = 4/3*pi* r**3
    return v
# Note: To change the value of r, you must change the r in the print line
print(sphere_volume(6))

#%%
#Problem 4
"""#A function that defines two matrices as NumPy arrays and returns
 the matrix product AB"""

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
""" A function that accepts a taxable income and returns the tax liability 
using 3 tax brackets """

def tax_liability(income):
    if income > 0 and income <9875:
        tax = income*0.1
        return tax
    elif income == 9875:
        tax = 987.5
        return tax
    elif income > 9875 and income < 40125:
        tax =987.5 + ((income - 9875) * 0.12)
        return tax
    elif income == 40125:
        tax = 4617.5
        return tax
    elif income > 40125:
        tax = 4617.5 + ((income - 40125)* 0.22)
        return tax
    else: 
        print("Lucky you, this script will lead to tax evasion!")


print('You owe', tax_liability(500000), 'in taxes')

#%%
#Problem 6A
""" A function that defines two vectors (A and B) as lists and returns their 
product, sum and 5*A. """


def prob6a():
    product = []
    add = []
    five = []
    A = [1, 2, 3, 4, 5, 6, 7]
    B = [5, 5, 5, 5, 5, 5, 5]
    for i, j in zip(A, B):
        product.append(i * j)
        add.append(i + j)
        five.append(5 * i)
    return(product, add, five)
    
print(prob6a())

#%%
#Problem 6B

""" A function that does the same as problem 6A, but utilizing numpy """

import numpy as np 

def prob6b():
    A = ([1, 2, 3, 4, 5, 6, 7])
    B = np.array([95, 5, 5, 5, 5, 5, 5])
    product = A*B
    add = A+B
    five = np.multiply(5, A)
    return(product, add, five)

print(prob6b())

    

