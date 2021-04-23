# Activation (sigmoid) function
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.dot(weights, features) + bias) 

# Error (log-loss) formula
def error_formula(y, output):
    return np.dot(-y, np.log(output))-np.dot((1-y), np.log(1-output)) 

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    y_hat = output_formula(x, weights, bias)
    weights = weights + np.dot(learnrate, np.dot((y-y_hat), x))
    bias = bias + learnrate*(y-y_hat)
    return weights, bias
