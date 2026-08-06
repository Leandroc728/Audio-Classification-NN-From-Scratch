import numpy as np

def update_parameters(parameters, grads, layers, learning_rate=0.01, lambd=0.0, m=None):
    ''' Update parameters using Gradient Descent with optional L2 regularization '''
    
    # Gets the number of layers
    L = len(layers)
    
    for l in range(1, L):
        dW = grads["dW" + str(l)]
        
        # Add L2 penalty to weight gradient
        if lambd > 0 and m is not None:
            dW = dW + (lambd / m) * parameters["W" + str(l)]
            
        parameters["W" + str(l)] -= learning_rate * dW
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]
        
    return parameters

def initialize_adam(parameters, layers):
    ''' Initializes moment vectors m and v as zeros matching parameter shapes '''
    
    # Initializes the variables
    L = len(layers)
    m = {}
    v = {}
    
    # For each layers initializes the m and v gradients as 0's
    for l in range(1, L):
        m["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        m["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        v["dW" + str(l)] = np.zeros_like(parameters["W" + str(l)])
        v["db" + str(l)] = np.zeros_like(parameters["b" + str(l)])
        
    return m, v

def update_parameters_adam(parameters, grads, m_adam, v_adam, t, layers, eta=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, lambd=0.0, m=None):
    ''' Updates parameters using Adam optimization with optional L2 regularization '''
    
    # Gets the number of layers
    L = len(layers)
    
    m_corrected = {}
    v_corrected = {}
    
    for l in range(1, L):
        dW = grads["dW" + str(l)]
        db = grads["db" + str(l)]
        
        # Add L2 Regularization penalty to weight gradient
        if lambd > 0 and m is not None:
            dW = dW + (lambd / m) * parameters["W" + str(l)]
        
        # Momentum
        m_adam["dW" + str(l)] = beta_1 * m_adam["dW" + str(l)] + (1 - beta_1) * dW
        m_adam["db" + str(l)] = beta_1 * m_adam["db" + str(l)] + (1 - beta_1) * db
        
        # Compute bias-corrected 1st moment estimate
        m_corrected["dW" + str(l)] = m_adam["dW" + str(l)] / (1 - beta_1**t)
        m_corrected["db" + str(l)] = m_adam["db" + str(l)] / (1 - beta_1**t)
        
        # RMSprop
        v_adam["dW" + str(l)] = beta_2 * v_adam["dW" + str(l)] + (1 - beta_2) * np.square(dW)
        v_adam["db" + str(l)] = beta_2 * v_adam["db" + str(l)] + (1 - beta_2) * np.square(db)
        
        # Compute bias-corrected 2nd moment estimate
        v_corrected["dW" + str(l)] = v_adam["dW" + str(l)] / (1 - beta_2**t)
        v_corrected["db" + str(l)] = v_adam["db" + str(l)] / (1 - beta_2**t)
        
        # Update parameters
        parameters["W" + str(l)] -= eta * (m_corrected["dW" + str(l)] / (np.sqrt(v_corrected["dW" + str(l)]) + epsilon))
        parameters["b" + str(l)] -= eta * (m_corrected["db" + str(l)] / (np.sqrt(v_corrected["db" + str(l)]) + epsilon))
        
    return parameters, m_adam, v_adam

def compute_decayed_learning_rate(eta_0, epoch, decay_rate=0.01, schedule="time_based", step_size=100):
    ''' Calculates the decayed learning rate based on the selected schedule '''
    
    if schedule == "time_based":
        return eta_0 / (1.0 + decay_rate * epoch)
        
    elif schedule == "exponential":
        return eta_0 * (decay_rate ** epoch)
        
    elif schedule == "step":
        return eta_0 * (decay_rate ** (epoch // step_size))
        
    return eta_0