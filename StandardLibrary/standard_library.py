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
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    list_1 = ["rock", "pop", "hip hop", "classical", "indie"]
    num_1 = 67583928
    string_1 = "I love pythons almost as much as vipers"
    tuple_1 = (5, 6, 'fun', 8, 'time', 64, 'potato')
    set_1 = {'beer', 'wine', 'whiskey', 'water'}
    if list_2 = list_1:
        list_2[2] = "R&B"
        print(list_1 == list_2)
    elif next():
        if num_2 = num_1:
            num_2[0] = 8
            print(num_1 == num_2)
        elif next():
            if string_2 = string_1:
                string_2[9] = "C"
                print(num_1 == num_2)
            elif next():
                if tuple_2 = tuple_1
            
        
    


#%%
"""
#%%
# Problem 3

from calculator import *

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
    sumsquares = calculator.sum(a**2, b**2)
    hypt = calculator.sqrt(sumsquares)
    return hypt

print(hypot(3, 4))
   
  #raise NotImplementedError("Problem 3 Incomplete")

#%%
# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    
    
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""