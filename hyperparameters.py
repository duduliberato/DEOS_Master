"""
Hyperparameter configurations for Machine Learning models in the forecasting system.

This module contains predefined lists of hyperparameters that will be tested
during model training and optimization.
"""

# Random Forest and XGBoost hyperparameters
NUMBER_OF_ESTIMATORS = [50, 100, 200, 300, 400, 500, 700, 1000]
MAXIMUM_DEPTH = [1, 2, 3, 6, 9, 12, 15, 20, 30]
LEARNING_RATE = [0.00001, 0.0001, 0.001, 0.01, 0.1]

# Neural Network layer configurations
LAYER_CONFIGURATIONS = [
    [30, 30],
    [50, 50],
    [100, 50, 50],
    [50, 100, 50],
    [50, 50, 100]
]

# Training hyperparameters
EPOCHS = [5, 10, 15, 20]
BATCH_SIZE = [256, 512, 1024, 2048]

# Additional hyperparameters that might be useful
MIN_SAMPLES_SPLIT = [2, 5, 10]
MIN_SAMPLES_LEAF = [1, 2, 4]
SUBSAMPLE = [0.8, 0.9, 1.0]
COL_SAMPLE_BYTREE = [0.8, 0.9, 1.0]

# Dropout rates for neural networks
DROPOUT_RATES = [0.1, 0.2, 0.3, 0.5]

# Activation functions for neural networks
ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'sigmoid']

# Optimizer learning rates for neural networks
OPTIMIZER_LEARNING_RATES = [0.001, 0.01, 0.1]

def get_hyperparameter_grid(model_type: str) -> dict:
    """
    Get a complete hyperparameter grid for a specific model type.
    
    Args:
        model_type (str): Type of model ('random_forest', 'xgboost', 'neural_network')
    
    Returns:
        dict: Dictionary containing hyperparameter lists for the specified model
    """
    if model_type.lower() == 'random_forest':
        return {
            'n_estimators': NUMBER_OF_ESTIMATORS,
            'max_depth': MAXIMUM_DEPTH
        }
    
    elif model_type.lower() == 'xgboost':
        return {
            'n_estimators': NUMBER_OF_ESTIMATORS,
            'max_depth': MAXIMUM_DEPTH,
            'learning_rate': LEARNING_RATE
        }
    
    elif model_type.lower() == 'lstm':
        return {
            'layer_configurations': LAYER_CONFIGURATIONS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rates': LEARNING_RATE
        }
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                       f"Supported types: 'random_forest', 'xgboost', 'lstm'")

def get_all_hyperparameters() -> dict:
    """
    Get all available hyperparameters in a single dictionary.
    
    Returns:
        dict: Dictionary containing all hyperparameter lists
    """
    return {
        'number_of_estimators': NUMBER_OF_ESTIMATORS,
        'maximum_depth': MAXIMUM_DEPTH,
        'learning_rate': LEARNING_RATE,
        'layer_configurations': LAYER_CONFIGURATIONS,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'min_samples_split': MIN_SAMPLES_SPLIT,
        'min_samples_leaf': MIN_SAMPLES_LEAF,
        'subsample': SUBSAMPLE,
        'col_sample_bytree': COL_SAMPLE_BYTREE,
        'dropout_rates': DROPOUT_RATES,
        'activation_functions': ACTIVATION_FUNCTIONS,
        'optimizer_learning_rates': OPTIMIZER_LEARNING_RATES
    }

# if __name__ == "__main__":
#     # Example usage
#     print("Available hyperparameters:")
#     print("=" * 50)
    
#     all_params = get_all_hyperparameters()
#     for param_name, param_values in all_params.items():
#         print(f"{param_name}: {param_values}")
    
#     print("\n" + "=" * 50)
#     print("Random Forest hyperparameters:")
#     print(get_hyperparameter_grid('random_forest'))
    
#     print("\nXGBoost hyperparameters:")
#     print(get_hyperparameter_grid('xgboost'))
    
#     print("\nNeural Network hyperparameters:")
#     print(get_hyperparameter_grid('neural_network'))
