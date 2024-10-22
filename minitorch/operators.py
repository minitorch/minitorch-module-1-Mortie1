"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(a: float, b: float) -> float:
    """Returns the product of `a` and `b`.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        Product of a and b

    """
    return a * b


# - id
def id(a: float) -> float:
    """Returns input as it is.

    Args:
    ----
        a: Float value

    Returns:
    -------
        `a` (input) as it is

    """
    return a


# - add
def add(a: float, b: float) -> float:
    """Returns the sum of `a` and `b`.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        Sum of a and b

    """
    return a + b


# - neg
def neg(a: float) -> float:
    """Returns negated input.

    Args:
    ----
        a: Float value

    Returns:
    -------
        Negated `a` (input)

    """
    return -a


# - lt
def lt(a: float, b: float) -> bool:
    """Returns whether `a` is less than `b`.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        True if `a` is less than `b`, false otherwise

    """
    return a < b


# - eq
def eq(a: float, b: float) -> bool:
    """Checks if two float values are equal.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        True if the two float values are equal, False otherwise.

    """
    return a == b


# - max
def max(a: float, b: float) -> float:
    """Returns the maximum of two float values.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        The maximum of `a` and `b`.

    """
    return a if lt(b, a) else b


# - is_close
def is_close(a: float, b: float) -> bool:
    """Checks if two float values are close.

    Args:
    ----
        a: First float value.
        b: Second float value.

    Returns:
    -------
        True if the two float values are close, False otherwise.

    """
    return abs(a - b) < 1e-2


# - sigmoid
def sigmoid(a: float) -> float:
    """Compute the sigmoid of `a`.

    Args:
    ----
        a: float value

    Returns:
    -------
        Sigmoid of `a`

    """
    if a >= 0:
        return 1.0 / (1.0 + math.exp(-a))
    return math.exp(a) / (1 + math.exp(a))


# - relu
def relu(a: float) -> float:
    """Compute the ReLU of `a`.

    Args:
    ----
        a: float value

    Returns:
    -------
        The ReLU of `a`

    """
    return max(0, a)


# - log
def log(a: float) -> float:
    """Compute the natural logarithm of `a`.

    Args:
    ----
        a: float value

    Returns:
    -------
        The natural logarithm of `a`

    """
    return math.log(a)


# - exp
def exp(a: float) -> float:
    """Compute the exponential of `a`.

    Args:
    ----
        a: float value

    Returns:
    -------
        The exponential of `a`

    """
    return math.exp(a)


# - log_back
def log_back(a: float, n: float) -> float:
    """Compute the derivative of the natural logarithm function with respect to its input.

    Args:
    ----
        a: The input float value for which the natural logarithm was computed.
        n: The upstream gradient value.

    Returns:
    -------
        The gradient of the logarithm function with respect to `a`, scaled by the upstream gradient `n`.

    """
    return (1 / a) * n


# - inv
def inv(a: float) -> float:
    """Compute the inverse of `a`.

    Args:
    ----
        a: float value

    Returns:
    -------
        The inverse of `a`

    """
    return 1 / a


# - inv_back
def inv_back(a: float, n: float) -> float:
    """Compute the derivative of the inverse function with respect to its input.

    Args:
    ----
        a: The input float value for which the inverse was computed.
        n: The upstream gradient value.

    Returns:
    -------
        The gradient of the inverse function with respect to `a`, scaled by the upstream gradient `n`.

    """
    return -1 / (a * a) * n


# - relu_back
def relu_back(a: float, n: float) -> float:
    """Compute the derivative of the ReLU function with respect to its input.

    Args:
    ----
        a: The input float value for which the ReLU was computed.
        n: The upstream gradient value.

    Returns:
    -------
        The gradient of the ReLU function with respect to `a`, scaled by the upstream gradient `n`.

    """
    return 1 * n if a > 0 else 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(f: Callable[..., float], a: Iterable) -> Iterable[float]:
    """Maps a function over an iterable. Given a function `f` and an iterable `a`, returns an iterable that
    contains the results of applying `f` to each element of `a`.

    Args:
    ----
        f: A function that returns a float.
        a: An iterable of floats

    Yields:
    ------
        An iterable of floats, where each element is the result of applying `f` to the corresponding element of `a`

    """
    for i in a:
        yield f(i)


def zipWith(a: Iterable[float], b: Iterable[float]) -> Iterable[tuple[float, float]]:
    """Zips two iterables together.

    Args:
    ----
        a: An iterable of floats.
        b: An iterable of floats.

    Yields:
    ------
        An iterable of tuples, where each tuple contains elements from the
        corresponding positions in `a` and `b`. Iteration stops when the shortest
        input iterable is exhausted.

    """
    a, b = iter(a), iter(b)
    while True:
        try:
            yield (next(a), next(b))
        except StopIteration:
            return


def reduce(f: Callable[[float, float], float], a: Iterable[float]) -> float:
    """Applies a binary function `f` to all items in `a`, going from left to right,
    so as to reduce the iterable to a single output. Given a binary function `f`
    and an iterable `a`, return the result of applying `f` to the first two elements of `a`,
    then applying `f` to the result and the next element, and so on until only one element remains.

    Args:
    ----
        f: A binary function that takes two floats as arguments and returns a float.
        a: An iterable of floats.

    Returns:
    -------
        The result of applying `f` to all elements of `a`, going from left to right.

    """
    start = True
    res = 0.0
    for i in a:
        if start:
            res = i
            start = False
        else:
            res = f(res, i)
    return res


def negList(a: Iterable[float]) -> Iterable[float]:
    """Negates each element in an iterable.

    Args:
    ----
        a: An iterable of floats.

    Returns:
    -------
        An iterable of floats, where each element is the negation of the corresponding element in `a`.

    """
    return map(neg, a)


def addLists(a: Iterable[float], b: Iterable[float]) -> Iterable[float]:
    """Adds two iterables together elementwise.

    Args:
    ----
        a: An iterable of floats.
        b: An iterable of floats.

    Returns:
    -------
        An iterable of floats, where each element is the sum of the corresponding elements in `a` and `b`.

    """
    return map(lambda x: add(x[0], x[1]), zipWith(a, b))


def sum(a: list) -> float:
    """Returns the sum of all elements in a list.

    Args:
    ----
        a: A list of floats.

    Returns:
    -------
        The sum of all elements in `a`.

    """
    return reduce(add, a)


def prod(a: list) -> float:
    """Returns the product of all elements in a list.

    Args:
    ----
        a: A list of floats.

    Returns:
    -------
        The product of all elements in `a`.

    """
    return reduce(mul, a)
