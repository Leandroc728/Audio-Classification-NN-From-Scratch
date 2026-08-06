import numpy as np

def initialize_parameters(layers):
    ''' Function for the initialization of parameters '''
    
    np.random.seed(42)
    
    # Initialize the parameters and get the number of layers in the Neural Network
    parameters = {}
    L = len(layers)
    
    for l in range(1, L):
        # He random initialization
        parameters["W" + str(l)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2.0 / layers[l - 1])
        parameters["b" + str(l)] = np.zeros((layers[l], 1))
    
    return parameters

def activation_function(z, activation="relu"):
    ''' Computes the activation function for input z '''

    # The two possible activation functions for this model
    # ReLU for the hidden layers and softmax for the output layer
    if activation == "relu":
        return np.maximum(0, z)
        
    elif activation == "softmax":
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)

def forward_propagation(X, parameters, layers):
    ''' Standard Foward Propagation '''
    
    # Varibles and parameters initialization
    caches = {}
    A = X
    caches["A0"] = X
    L = len(layers)
    
    # For each layer compute the forward pass process
    for l in range(1, L):
        A_prev = A
        
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A_prev) + b
        
        # Check if is the final layer, if so changes the activation to "softmax"
        if l < L - 1:
            A = activation_function(Z, activation="relu")
        else:
            A = activation_function(Z, activation="softmax")

        caches["Z" + str(l)] = Z
        caches["A" + str(l)] = A
    
    AL = A
    return AL, caches

# Dropout alternatives
def forward_propagation_with_dropout(X, parameters, layers, keep_prob=0.8, is_training=True):
    ''' Forward propagation with Inverted Dropout on hidden layers '''
    
    # Varibles and parameters initialization 
    caches = {}
    A = X
    caches["A0"] = X
    L = len(layers)
    
    # For each layer compute the forward pass with dropout for the hidden layers
    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A_prev) + b
        
        # Check if is the final layer, if so changes the activation to "softmax"
        if l < L - 1:
            A = activation_function(Z, activation="relu")
            
            # Dropout
            if is_training and keep_prob < 1.0:
                D = np.random.rand(*A.shape) < keep_prob
                
                A = (A * D) / keep_prob
                caches["D" + str(l)] = D
        else:
            A = activation_function(Z, activation="softmax")

        caches["Z" + str(l)] = Z
        caches["A" + str(l)] = A
    
    AL = A
    return AL, caches

def backward_propagation(Y, caches, parameters, layers):
    ''' Standard Backward Propagation '''
    
    # Varibles and parameters initialization
    grads = {}
    L = len(layers)
    m = Y.shape[1]
    
    # Gets the caches and calculates the loss in the last layer
    AL = caches["A" + str(L - 1)]
    dZ = AL - Y
    
    # For each layer compute the gradients
    for l in reversed(range(1, L)):
        A_prev = caches["A" + str(l - 1)]
        
        grads["dW" + str(l)] = (1.0 / m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l)] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            W = parameters["W" + str(l)]
            dA_prev = np.dot(W.T, dZ)
            Z_prev = caches["Z" + str(l - 1)]
            
            dZ = dA_prev * (Z_prev > 0)
            
    return grads

def backward_propagation_with_dropout(Y, caches, parameters, layers, keep_prob=0.8):
    ''' Backward propagation accounting for dropout masks stored during forward pass '''
    
    # Varibles and parameters initialization
    grads = {}
    L = len(layers)
    m = Y.shape[1]
    
    # Gets the caches and calculates the loss in the last layer
    AL = caches["A" + str(L - 1)]
    dZ = AL - Y
    
    # For each layers computes the gradients considering the dropout
    for l in reversed(range(1, L)):
        A_prev = caches["A" + str(l - 1)]
        
        grads["dW" + str(l)] = (1.0 / m) * np.dot(dZ, A_prev.T)
        grads["db" + str(l)] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        
        if l > 1:
            W = parameters["W" + str(l)]
            dA_prev = np.dot(W.T, dZ)
            
            # Dropout consideration
            if keep_prob < 1.0:
                D_prev = caches["D" + str(l - 1)]
                dA_prev = (dA_prev * D_prev) / keep_prob
            
            Z_prev = caches["Z" + str(l - 1)]
            dZ = dA_prev * (Z_prev > 0)
            
    return grads