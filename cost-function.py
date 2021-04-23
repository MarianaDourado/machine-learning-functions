def costFunction(theta, X, y):
    """
    Compute cost and gradient for logistic regression. 
    
    Parameters
    ----------
    theta : array_like
        The parameters for logistic regression. This a vector
        of shape (n+1, ).
    
    X : array_like
        The input dataset of shape (m x n+1) where m is the total number
        of data points and n is the number of features. We assume the 
        intercept has already been added to the input.
    
    y : arra_like
        Labels for the input. This is a vector of shape (m, ).
    
    Returns
    -------
    J : float
        The computed value for the cost function. 
    
    grad : array_like
        A vector of shape (n+1, ) which is the gradient of the cost
        function with respect to theta, at the current values of theta.
        
    Instructions
    ------------
    Compute the cost of a particular choice of theta. You should set J to 
    the cost. Compute the partial derivatives and set grad to the partial
    derivatives of the cost w.r.t. each parameter in theta.
    """
    # Initialize some useful values
    m = y.size  # number of training examples

    # You need to return the following variables correctly 
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    h = sigmoid(np.dot(X, theta))
    grad = (1/m) * (np.dot((h - y), X))
    """ LEMBRE-SE:
    O número de colunas do primeiro vetor deve ser igual ao numero de linhas do segundo vetor.
    O resultado da operação será um vetor com o mesmo numero de linhas do primeiro e o mesmo numero de colunas do segundo.
    """
    J = (1/m) * np.sum((np.dot(-y, np.log(h))) - ((np.dot(1-y, np.log(1-h)))))
    # =============================================================
    return J, grad
