import numpy as np
from .layers import initialize_parameters, forward_propagation, forward_propagation_with_dropout, backward_propagation, backward_propagation_with_dropout
from .loss import compute_loss
from .optimizers import initialize_adam, update_parameters_adam, compute_decayed_learning_rate, update_parameters

class DenseNeuralNetwork:
    ''' Dense Neural Network Class definition '''
    
    def __init__(self, layers):
        self.layers = layers
        self.parameters = initialize_parameters(layers)
    
    def fit(self, X, Y, epochs=500, learning_rate=0.01):
        ''' Standard training loop '''
        
        history = []
        
        for epoch in range(1, epochs + 1):
            AL, caches = forward_propagation(X, self.parameters, self.layers)
            
            loss = compute_loss(Y, AL)
            
            # Keeps track of the loss
            history.append(loss)
            
            grads = backward_propagation(Y, caches, self.parameters, self.layers)
            
            self.parameters = update_parameters(self.parameters, grads, self.layers, learning_rate)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}\nLoss: {loss:.4f}\nLR: {learning_rate:.6f}")
            
        return history
                
    def fit_optimized(self, X, Y, epochs=500, eta_0=0.01, keep_prob=0.8, decay_rate=0.005, lambd=0.01):
        ''' Optimized training loop using Adam, Dropout, LR Decay and L2 Regularization '''
        
        history = []

        # Gets the number of examples and initializes adam vectors
        m_samples = X.shape[1]
        m_adam, v_adam = initialize_adam(self.parameters, self.layers)
        
        for epoch in range(1, epochs + 1):
            eta = compute_decayed_learning_rate(eta_0, epoch, decay_rate=decay_rate)
            
            AL, caches = forward_propagation_with_dropout(X, self.parameters, self.layers, keep_prob=keep_prob, is_training=True)
            
            loss = compute_loss(Y, AL, parameters=self.parameters, lambd=lambd)
            
            history.append(loss)
            
            grads = backward_propagation_with_dropout(Y, caches, self.parameters, self.layers, keep_prob=keep_prob)
            
            self.parameters, m_adam, v_adam = update_parameters_adam(self.parameters, grads, m_adam, v_adam, t=epoch, layers=self.layers, eta=eta, lambd=lambd, m=m_samples)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}\nLoss: {loss:.4f}\n LR: {eta:.6f}")
        
        return history
                
    def predict(self, X):
        ''' Prediction function '''
        
        AL, _ = forward_propagation_with_dropout(X, self.parameters, self.layers, keep_prob=1.0, is_training=False)
        
        return np.argmax(AL, axis=0)