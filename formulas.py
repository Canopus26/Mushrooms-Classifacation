import math

def sig(x):
    #calculation of logistic function as activation function
    a = 1.0
    b = 1.0 + math.exp(-x)
    return a/b

def inv_sig(x):
    #calculation of derivative of the output of node with respect to its input
    c = sig(x)
    return c * (1 - c)

def err(o, t):
    #squared error function, o is the real output value and t is the target output 
    err = t-o
    return 0.5 * (math.pow(err,2))

def inv_err(o, t):
    #difference between real output value and target output value
    inv_err = o-t
    return inv_err