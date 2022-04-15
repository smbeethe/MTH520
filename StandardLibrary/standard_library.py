# standard_library.py
"""Python Essentials: The Standard Library.
Sarah Beethe
MTH 520
4/15/2022
"""

#%%
# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """

    return min(L), max(L), sum(L)/len(L)

    raise NotImplementedError("Problem 1 Incomplete")

L = [1, 2, 3, 4, 5, 6, 7]
print(prob1(L))

#%%
# Problem 2
import sys

def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    list_1 = ["rock", "pop", "hip hop", "classical", "indie"]
    list_2 = list_1
    list_2[2] = "R&B"
    print(list_2 == list_1)
    if list_2 == list_1:
        print("Lists are mutable")
    else:
        print("lists are immutable")
    
    num_1 = 67583928
    num_2 = num_1
    num_2 = 7
    print(num_2 == num_1)
    if num_2 == num_1:
        print("Integers are mutable")
    else:
        print("Integers are immutable")
    
    string_1 = "I love pythons almost as much as vipers"
    string_2 = string_1
    string_2 = "C"
    print(string_2 == string_1)
    if string_2 == string_1:
        print("Strings are mutable")
    else:
        print("Strings are immutable")
    
    tuple_1 = (5, 6, 'fun', 8, 'time', 64, 'potato')
    tuple_2 = tuple_1
    tuple_2 = (3, 4, 5)
    print(tuple_2 == tuple_1)
    if tuple_2 == tuple_1:
        print("Tuples are mutable")
    else:
        print("Tuples are immutable")
    
    set_1 = {'beer', 'wine', 'whiskey','water'}
    set_2 = set_1
    set_2.add(4)
    print(set_2 == set_1)
    if set_2 == set_1:
        print("Sets are mutable")
    else:
        print("Sets are immutable")
    
    sys.exit()
    return prob2()

#Had to add the sys.exit() line because the function was recursive and sysexit is the easiest fix
prob2()

#%%

# Problem 3

#As discussed in Lab today... this script cannot import calculator
#even though it should be able to The program works with the functions
#used in the calcuator.py script in the StandardLibrary folder, which have
#been commented out, but can be commented in to show that this should work

import calculator as calc
import math

#def sum(a, b):
#    return a+b
#
#def product(a, b):
#    return a*b

def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    sumsquares = calc.sum(a**2, b**2)
    hyp = math.sqrt(sumsquares)
    
    return hyp

    raise NotImplementedError("Problem 3 Incomplete")

print(hypot(3, 4))
   
  

#%%
# Problem 4

from itertools import chain, combinations

def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    #A = {1, 2, 3}
    return chain.from_iterable(combinations(iterable, r) for r in range(len(s)+1)

    
#    raise NotImplementedError("Problem 4 Incomplete")

A = {1, 2, 3, 4, 5}
print(power_set(A))

#%%
#Don't do this one!
# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""