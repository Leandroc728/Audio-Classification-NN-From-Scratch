import numpy as np
from engine.layers import initialize_parameters, activation_function

def test_initialize_parameters():
    ''' Ensure weight and bias matrices are initialized with correct dimensions '''
    
    # Define the architecture and initialize parameters
    layers = [40, 16, 5]
    parameters = initialize_parameters(layers)
    
    # Verify that all expected weight and bias matrices exist
    assert "W1" in parameters
    assert "b1" in parameters
    assert "W2" in parameters
    assert "b2" in parameters
    
    # Check the dimensions of the first layer parameters
    assert parameters["W1"].shape == (16, 40)
    assert parameters["b1"].shape == (16, 1)
    
    # Check the dimensions of the output layer parameters
    assert parameters["W2"].shape == (5, 16)
    assert parameters["b2"].shape == (5, 1)

def test_relu_activation():
    ''' Ensure ReLU correctly zeroes out negative numbers and passes positive ones '''
    
    # Create an input containing negative, zero, and positive values and apply the ReLU activation
    z = np.array([[-1.0, 0.0, 2.5, -5.0]])
    a = activation_function(z, activation="relu")
    
    # Expected output
    expected = np.array([[0.0, 0.0, 2.5, 0.0]])
    
    # Verify that the activation output matches the expected values
    np.testing.assert_array_equal(a, expected)

def test_softmax_activation():
    ''' Ensure Softmax outputs sum to 1.0 down the column '''
    
    # Create sample logits for two examples and three classes
    z = np.array([
        [1.0, 2.0],
        [3.0, 3.0],
        [1.0, 2.0]
    ])
    
    # Apply the Softmax activation function
    a = activation_function(z, activation="softmax")
    
    assert a.shape == (3, 2)
    
    # Compute the sum of probabilities for each example
    col_sums = np.sum(a, axis=0)
    
    #  Verify that each column sums to 1, as required by Softmax
    np.testing.assert_allclose(col_sums, np.array([1.0, 1.0]))