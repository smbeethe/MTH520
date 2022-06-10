# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Sarah Beethe>
<MTH 520>
<5/19/22>
"""

# %%
# Problem 1

import numpy as np
from matplotlib import pyplot as plt


def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    matrix = np.array(np.random.normal(size=(n, n)))
    means = np.array(matrix.mean(axis=1))
   # print(matrix)
   # print(means)
    return np.var(means)

    raise NotImplementedError("Problem 1 Incomplete")


print(var_of_means(1000))


def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    ar = []
    for i in range(100, 1100, 100):
        var = var_of_means(i)
        ar.append(var)
        var_ar = np.array(ar)
        plt.plot(var_ar)
    plt.show()
    print(var_ar)


prob1()


# %%
# Problem 2

def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    c = plt.plot(x, np.cos(x))
    s = plt.plot(x, np.sin(x))
    t = plt.plot(x, np.arctan(x))
    plt.show()
    print(c, s, t)


prob2()
# %%
# Problem 3


def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 1, 100)
    x2 = np.linspace(1, 6, 100)
    f1 = 1/(x1-1)
    f2 = 1/(x2-1)
    plt.plot(x1, f1, 'm--', linewidth=4)
    plt.plot(x2, f2, 'm--', linewidth=4)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()


prob3()
# %%
# Problem 4


def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """

    x = np.linspace(0, 2*np.pi, 100)

    x1 = plt.subplot(221)
    x1.plot(x, np.sin(x), 'g')
    plt.title("sin(x)")
    plt.axis([0, 2*np.pi, -2, 2])

    x2 = plt.subplot(222)
    x2.plot(x, np.sin(2*x), "r--")
    x2.set_title("sin(2x)")
    plt.axis([0, 2*np.pi, -2, 2])

    x3 = plt.subplot(223)
    x3.plot(x, 2*np.sin(x), "b--")
    x3.set_title("2sin(x)")
    plt.axis([0, 2*np.pi, -2, 2])

    x4 = plt.subplot(224)
    x4.plot(x, 2*np.sin(2*x), "m:")
    x4.set_title("2sin(2x)")
    plt.axis([0, 2*np.pi, -2, 2])

    plt.suptitle("Variation of sin functions")
    plt.tight_layout(pad=0.4)

    plt.show()


prob4()


# %%
# Problem 6

import numpy as np
from matplotlib import pyplot as plt

def prob6():
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.copy(x)
    X, Y = np.meshgrid(x, y)
    Z = ((np.sin(X) * np.sin(Y))/(X*Y))

    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap = "viridis")
    plt.colorbar()
    
    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap = "coolwarm")
    plt.colorbar()
    
    plt.suptitle("Problem 6")
    plt.tight_layout(pad=0.4)
    
    plt.show()

prob6()
