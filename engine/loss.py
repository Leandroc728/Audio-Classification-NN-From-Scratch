import numpy as np

def compute_loss(Y, Y_hat, parameters=None, lambd=0.0):
    ''' Compute categorical cross-entropy cost with optional L2 penalty '''
    
    # Gets the number of examples and formats the prediction
    m = Y.shape[1]
    Y_hat = np.clip(Y_hat, 1e-15, 1.0 - 1e-15)
    
    cross_entropy_cost = - (1.0 / m) * np.sum(Y * np.log(Y_hat))
    
    # Add L2 penalty cost if parameters and lambd are provided
    if lambd > 0 and parameters is not None:
        l2_cost = 0
        
        L = len(parameters) // 2 + 1
        
        for l in range(1, L):
            l2_cost += np.sum(np.square(parameters["W" + str(l)]))
        
        l2_cost = (lambd / (2 * m)) * l2_cost
        
        return np.squeeze(cross_entropy_cost + l2_cost)
        
    return np.squeeze(cross_entropy_cost)