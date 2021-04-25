import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    """
    Softmax function transforms input values into values between 0 and 1, so that they can be interpreted as probabilities.
    """
    expL = np.exp(L) #"np.exp" - Calculate the exponential of all elements in the input array.
    result = []
    for i in expL:
        result.append(i/sum(expL))
    return result
    """ Better way:
    expL = np.exp(L)
    return np.divide(expL, expL.sum())
    """
