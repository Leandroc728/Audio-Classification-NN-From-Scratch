import numpy as np
from pipeline.audio_processor import process_audio_and_extract_mfcc
from engine.model import DenseNeuralNetwork

def train_val_split(X, Y, split_ratio=0.8, seed=42):
    ''' Shuffles dataset columns and splits into Train and Validation sets '''
    
    np.random.seed(seed)
    m = X.shape[1]
    
    # Shuffle column indices
    permutation = np.random.permutation(m)
    X_shuffled = X[:, permutation]
    Y_shuffled = Y[:, permutation]
    
    # Compute the number of training examples
    train_size = int(m * split_ratio)
    
     # Split the shuffled data into training and validation sets
    X_train = X_shuffled[:, :train_size]
    Y_train = Y_shuffled[:, :train_size]
    X_val = X_shuffled[:, train_size:]
    Y_val = Y_shuffled[:, train_size:]
    
    return X_train, Y_train, X_val, Y_val

def main():
    ''' Main Function '''
    
    # Load Audio Features
    X, Y = process_audio_and_extract_mfcc()
    
    # Split Dataset 80/20
    X_train, Y_train, X_val, Y_val = train_val_split(X, Y, split_ratio=0.8)
    
    # Determine the input dimension and number of output classes
    input_dim = X_train.shape[0] 
    num_classes = Y_train.shape[0] 
    
    # Define Model Architecture
    layers = [input_dim, 32, 16, num_classes]
    
    # Instantiate Model
    model = DenseNeuralNetwork(layers)
    
    # Train Model
    model.fit_optimized(
        X_train, 
        Y_train, 
        epochs=600, 
        eta_0=0.005, 
        keep_prob=0.7,
        decay_rate=0.001,
        lambd=0.05
    )
    
    # Predict the labels of the training set
    train_preds = model.predict(X_train)
    
    # Convert one-hot encoded labels to class indices
    train_targets = np.argmax(Y_train, axis=0)
    
    # Compute the training accuracy
    train_acc = np.mean(train_preds == train_targets) * 100

    # Predict the labels of the validation set
    val_preds = model.predict(X_val)
    
    # Convert one-hot encoded validation labels to class indices
    val_targets = np.argmax(Y_val, axis=0)
    
    # Compute the validation accuracy
    val_acc = np.mean(val_preds == val_targets) * 100

    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Val Accuracy:   {val_acc:.2f}%")
    
    # Save Trained Parameters to Disk
    np.savez("trained_model_params.npz", **model.parameters)

if __name__ == "__main__":
    main()