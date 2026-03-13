import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import time
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import itertools
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from hyperparameters import get_hyperparameter_grid

# === PLOTTING CONFIGURATION ===

def verify_file(country: str, chosen_model: str) -> tuple:
    """
    Verify if the country and model folders exist, create CSV files if needed.
    
    Args:
        country (str): Country name (e.g., "Brazil", "Portugal")
        chosen_model (str): Model type ("Random_Forest", "XGBoost", "LSTM")
    
    Returns:
        tuple: (results_path, plot_dir, csv_exists)
    """
    # Validate model type
    valid_models = ["Random_Forest", "XGBoost", "LSTM"]
    if chosen_model not in valid_models:
        raise ValueError(f"Invalid model. Must be one of: {valid_models}")
    
    # Create paths
    results_base = f"Results/{country}/{chosen_model}"
    results_path = f"{results_base}/results.csv"
    plot_dir = f"{results_base}/Plots"
    
    # Create directories if they don't exist
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Check if CSV exists and create headers if needed
    csv_exists = os.path.exists(results_path)
    
    if not csv_exists:
        # Create CSV with appropriate headers based on model
        if chosen_model == "Random_Forest":
            headers = ['mode', 'n_features', 'max_depth', 'n_estimators', 
                      'rmse_percentual_train', 'mape_train', 'rmse_percentual_test', 
                      'mape_test', 'processing_time']
        elif chosen_model == "XGBoost":
            headers = ['mode', 'n_features', 'max_depth', 'learning_rate', 'n_estimators',
                      'rmse_percentual_train', 'mape_train', 'rmse_percentual_test', 
                      'mape_test', 'processing_time']
        elif chosen_model == "LSTM":
            headers = ['mode', 'n_features', 'layers', 'learning_rate', 'epochs', 'batch_size',
                      'rmse_percentual_train', 'mape_train', 'rmse_percentual_test', 
                      'mape_test', 'processing_time']
        
        # Create empty CSV with headers
        df_empty = pd.DataFrame(columns=headers)
        df_empty.to_csv(results_path, index=False)
        print(f"✅ Created new CSV file: {results_path}")
    
    return results_path, plot_dir, csv_exists

def several_simulations(results_path: str, chosen_model: str, df_train: pd.DataFrame, 
                       df_test: pd.DataFrame) -> None:
    """
    Run multiple simulations for the specified model.
    
    Args:
        results_path (str): Path to save results CSV
        chosen_model (str): Model type ("Random_Forest", "XGBoost", "LSTM")
        df_train (pd.DataFrame): Training data
        df_test (pd.DataFrame): Test data
        mode (str): Mode type ("calendar" or "Fourier Ramps")
    """
    
    # Feature setup
    exclude_cols = {'time', 'timestamp', 'value'}
    features = [c for c in df_train.columns if c not in exclude_cols]
    
    # Ensure mode is exclusive: it must be either "calendar" or "Fourier Ramps"
    mode = detect_mode(features)
    
    if chosen_model == "Random_Forest":
        run_random_forest_simulations(results_path, df_train, df_test, features, mode)
    elif chosen_model == "XGBoost":
        run_xgboost_simulations(results_path, df_train, df_test, features, mode)
    elif chosen_model == "LSTM":
        run_lstm_simulations(results_path, df_train, df_test, features, mode)
    else:
        raise ValueError(f"Unsupported model: {chosen_model}")

def plot_simulation_results(results_path: str, chosen_model: str, country: str, nationality: str, time_limit: float) -> None:
    """
    Management function to dispatch plotting tasks based on the chosen model.

    Args:
        results_path (str): The file path to the CSV results file.
        chosen_model (str): The model to plot results for ("Random_Forest", "XGBoost", "LSTM").
        country (str): The country name for plot titles.
        nationality (str): The nationality name for plot titles.
    """
    model_plotters = {
        "Random_Forest": plot_tree_model_results,
        "XGBoost": plot_tree_model_results,
        "LSTM": plot_lstm_results,
    }

    if chosen_model in model_plotters:
        plotter_func = model_plotters[chosen_model]
        plotter_func(results_path, chosen_model, country, nationality, time_limit)
    else:
        raise ValueError(f"Unsupported model for plotting: {chosen_model}")

def detect_mode(features: list):

    has_hour = 'hour' in features
    has_fft  = 'fft_24h_signal' in features

    if has_hour and has_fft:
        return "Calendar and Fourier Ramps"
    elif has_hour:
        return "Calendar"
    elif has_fft:
        return "Fourier Ramps"
    else:
        return "Unknown"
    
def model_forecast(model_name: str, **kwargs) -> dict:
    """
    Acts as a controller to run a forecast for a specific model.

    Args:
        model_name (str): The name of the model ("Random_Forest", "XGBoost", "LSTM").
        **kwargs: A dictionary of arguments required by the specific model's
                  forecasting function (e.g., df_train, features, nf, depth, etc.).

    Returns:
        A dictionary containing the results of the forecast.
    """
    # Map model names to their respective forecasting functions
    model_dispatch = {
        "Random_Forest": random_forest_forecast,
        "XGBoost": xgboost_forecast,
        "LSTM": lstm_forecast,
    }

    # Get the correct function from the map
    forecast_func = model_dispatch.get(model_name)

    if forecast_func:
        # Call the function, passing all the keyword arguments
        return forecast_func(**kwargs)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# def run_random_forest_simulations(results_path: str, df_train: pd.DataFrame, 
#                                 df_test: pd.DataFrame, features: list, mode: str) -> None:
#     """Run Random Forest simulations with hyperparameter grid search."""
    
#     # Get hyperparameters from hyperparameters.py
#     hp_grid = get_hyperparameter_grid('random_forest')
#     max_depths = hp_grid['max_depth']
#     n_estimators = hp_grid['n_estimators']
    
#     df_existing = pd.read_csv(results_path)
#     existing_tuples = set(zip(df_existing['n_features'], 
#                             df_existing['max_depth'], 
#                             df_existing['n_estimators'],
#                             df_existing['mode'])) 


#     # Generate all possible combinations
#     full_tuples = {
#         (nf, d, ne, mode)
#         for nf in range(1, len(features) + 1)
#         for d in max_depths
#         for ne in n_estimators
#     }
    
#     missing = full_tuples - existing_tuples
#     results_new = []
    
#     print(f"Running {len(missing)} Random Forest simulations...")
    
#     for nf, depth, ne, mod in sorted(missing):
#         subset = features[:nf]
#         X_sub_tr = df_train[subset].values
#         X_sub_te = df_test[subset].values
#         y_tr = df_train['value'].values.ravel()
#         y_te = df_test['value'].values.ravel()
        
#         t0 = time.perf_counter()
#         model = RandomForestRegressor(
#             max_depth=depth,
#             n_estimators=ne,
#             bootstrap=True,
#             oob_score=True,
#             n_jobs=mp.cpu_count() // 2,
#             warm_start=True
#         )
#         model.fit(X_sub_tr, y_tr)
#         elapsed = time.perf_counter() - t0
        
#         # Predictions
#         y_pred_train = model.predict(X_sub_tr)
#         y_pred_test = model.predict(X_sub_te)
        
#         # Metrics
#         train_rmse_pct = 100 * np.sqrt(mean_squared_error(y_tr, y_pred_train)) / np.mean(y_tr)
#         train_mape = 100 * np.mean(np.abs((y_tr - y_pred_train) / y_tr))
#         test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te, y_pred_test)) / np.mean(y_te)
#         test_mape = 100 * np.mean(np.abs((y_te - y_pred_test) / y_te))
        
#         results_new.append({
#             'mode': mod,
#             'n_features': nf,
#             'max_depth': depth,
#             'n_estimators': ne,
#             'rmse_percentual_train': train_rmse_pct,
#             'mape_train': train_mape,
#             'rmse_percentual_test': test_rmse_pct,
#             'mape_test': test_mape,
#             'processing_time': elapsed
#         })
        
#         print(f"RF: nf={nf}, depth={depth}, ne={ne}, "
#               f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
#               f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
#               f"time={elapsed:.1f}s")
    
#     # Save new results
#     if results_new:
#         df_new = pd.DataFrame(results_new)
#         df_new.to_csv(results_path, mode='a', header=False, index=False)
#         print(f"✅ Saved {len(results_new)} Random Forest results")

def run_random_forest_simulations(results_path, df_train, df_test, features, mode):
    """Manages the simulation runs for the Random Forest model."""
    
    # 1. Define Hyperparameters
    hp_grid = get_hyperparameter_grid('random_forest') # Assumes this function exists
    max_depths = hp_grid['max_depth']
    n_estimators = hp_grid['n_estimators']
    
    # 2. Find Missing Simulations
    df_existing = pd.read_csv(results_path)
    existing_tuples = set(zip(
        df_existing['n_features'], 
        df_existing['max_depth'], 
        df_existing['n_estimators'],
        df_existing['mode']
    ))

    full_tuples = {
        (nf, d, ne, mode)
        for nf in range(1, len(features) + 1)
        for d in max_depths
        for ne in n_estimators
    }
    
    missing = full_tuples - existing_tuples
    if not missing:
        print("✅ No new Random Forest simulations to run.")
        return

    print(f"Running {len(missing)} new Random Forest simulations...")
    
    for nf, depth, ne, mod in sorted(missing):
        forecast_args = {
            'df_train': df_train,
            'df_test': df_test,
            'features': features,
            'mod': mod,
            'nf': nf,
            'depth': depth,
            'ne': ne
        }
        
        # Call the main controller function to get a single result
        result, y_pred_test = model_forecast("Random_Forest", **forecast_args)

        # --- START OF THE FIX ---
        # Convert the single result dictionary to a DataFrame
        df_new = pd.DataFrame([result])
        
        # Check if the file needs a header (only for the very first write)
        header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
        
        # Append the single result to the CSV file
        df_new.to_csv(results_path, mode='a', header=header, index=False)
        # --- END OF THE FIX ---


# def run_xgboost_simulations(results_path: str, df_train: pd.DataFrame, 
#                            df_test: pd.DataFrame, features: list, mode: str) -> None:
#     """Run XGBoost simulations with hyperparameter grid search."""
    
#     # Get hyperparameters from hyperparameters.py
#     hp_grid = get_hyperparameter_grid('xgboost')
#     max_depths = hp_grid['max_depth']
#     n_estimators = hp_grid['n_estimators']
#     learning_rates = hp_grid['learning_rate']
    
#     # Load existing results
#     df_existing = pd.read_csv(results_path)
#     existing_tuples = set(zip(df_existing['n_features'],
#                                 df_existing['max_depth'], 
#                                 df_existing['n_estimators'], 
#                                 df_existing['learning_rate'], 
#                                 df_existing['mode'])) 


#     # Generate all possible combinations
#     full_tuples = {
#         (nf, d, ne, lr, mode)
#         for nf in range(1, len(features) + 1)
#         for d in max_depths
#         for ne in n_estimators
#         for lr in learning_rates
#     }
    
#     missing = full_tuples - existing_tuples
#     results_new = []
    
#     print(f"Running {len(missing)} XGBoost simulations...")
    
#     for nf, depth, ne, lr, mod in sorted(missing):
#         subset = features[:nf]
#         X_sub_tr = df_train[subset].values
#         X_sub_te = df_test[subset].values
#         y_tr = df_train['value'].values.ravel()
#         y_te = df_test['value'].values.ravel()
        
#         t0 = time.perf_counter()
#         model = xgb.XGBRegressor(
#             max_depth=depth,
#             n_estimators=int(ne),
#             learning_rate=lr,
#             bootstrap=True,
#             n_jobs=mp.cpu_count() // 2,
#             tree_method="hist",
#             verbosity=0,
#             random_state=42
#         )
#         model.fit(X_sub_tr, y_tr)
#         elapsed = time.perf_counter() - t0
        
#         # Predictions
#         y_pred_train = model.predict(X_sub_tr)
#         y_pred_test = model.predict(X_sub_te)
        
#         # Metrics
#         train_rmse_pct = 100 * np.sqrt(mean_squared_error(y_tr, y_pred_train)) / np.mean(y_tr)
#         train_mape = 100 * np.mean(np.abs((y_tr - y_pred_train) / y_tr))
#         test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te, y_pred_test)) / np.mean(y_te)
#         test_mape = 100 * np.mean(np.abs((y_te - y_pred_test) / y_te))
        
#         results_new.append({
#             'mode': mod,
#             'n_features': nf,
#             'max_depth': depth,
#             'learning_rate': lr,
#             'n_estimators': ne,
#             'rmse_percentual_train': train_rmse_pct,
#             'mape_train': train_mape,
#             'rmse_percentual_test': test_rmse_pct,
#             'mape_test': test_mape,
#             'processing_time': elapsed
#         })
        
#         print(f"XGB: nf={nf}, depth={depth}, lr={lr}, ne={ne}, "
#               f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
#               f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
#               f"time={elapsed:.1f}s")
    
#     # Save new results
#     if results_new:
#         df_new = pd.DataFrame(results_new)
#         df_new.to_csv(results_path, mode='a', header=False, index=False)
#         print(f"✅ Saved {len(results_new)} XGBoost results")

def run_xgboost_simulations(results_path, df_train, df_test, features, mode):
    """Manages the simulation runs for the XGBoost model."""

    # 1. Define Hyperparameters
    hp_grid = get_hyperparameter_grid('xgboost') # Assumes this function exists
    max_depths = hp_grid['max_depth']
    n_estimators = hp_grid['n_estimators']
    learning_rates = hp_grid['learning_rate']

    # 2. Find Missing Simulations
    df_existing = pd.read_csv(results_path)
    existing_tuples = set(zip(
        df_existing['n_features'], 
        df_existing['max_depth'], 
        df_existing['n_estimators'],
        df_existing['learning_rate'],
        df_existing['mode']
    ))

    full_tuples = {
        (nf, d, ne, lr, mode)
        for nf in range(1, len(features) + 1)
        for d in max_depths
        for ne in n_estimators
        for lr in learning_rates
    }
    
    missing = full_tuples - existing_tuples
    if not missing:
        print("✅ No new XGBoost simulations to run.")
        return
        
    print(f"Running {len(missing)} new XGBoost simulations...")

    # 2. Run simulations and save each result immediately after it's generated
    for nf, depth, ne, lr, mod in sorted(missing):
        forecast_args = {
            'df_train': df_train,
            'df_test': df_test,
            'features': features,
            'mod': mod,
            'nf': nf,
            'depth': depth,
            'ne': ne,
            'lr': lr
        }
        
        # Get a single result from the controller function
        result, y_pred_test = model_forecast("XGBoost", **forecast_args)

        # --- START OF THE FIX ---
        # Convert the single result dictionary into a DataFrame
        df_new = pd.DataFrame([result])
        
        # Check if the CSV file needs a header (for the very first write)
        header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
        
        # Append the single-row DataFrame to the CSV file
        df_new.to_csv(results_path, mode='a', header=header, index=False)
        # --- END OF THE FIX ---

# def run_lstm_simulations(results_path: str, df_train: pd.DataFrame, 
#                         df_test: pd.DataFrame, features: list, mode: str) -> None:
#     """Run LSTM simulations with hyperparameter grid search."""
    
#     # Get hyperparameters from hyperparameters.py
#     hp_grid = get_hyperparameter_grid('lstm')
#     layer_configs = hp_grid['layer_configurations']
#     epochs_list = hp_grid['epochs']
#     batch_sizes = hp_grid['batch_size']
#     learning_rates = hp_grid['learning_rates']
    
#     # Load existing results
#     df_existing = pd.read_csv(results_path)
#     existing_tuples = set()
#     for _, row in df_existing.iterrows():
#         try:
#             layers = eval(row['layers']) if isinstance(row['layers'], str) else row['layers']
#             existing_tuples.add((
#                 row['n_features'], 
#                 tuple(layers), 
#                 row['epochs'], 
#                 row['batch_size'], 
#                 row['learning_rate'],
#                 row['mode']
#             ))
#         except:
#             continue
    
#     # Generate all possible combinations
#     full_tuples = {
#         (nf, tuple(layers), epochs, batch_size, lr, mode)
#         for nf in range(1, len(features) + 1)
#         for layers in layer_configs
#         for epochs in epochs_list
#         for batch_size in batch_sizes
#         for lr in learning_rates
#     }
    
#     missing = full_tuples - existing_tuples
#     results_new = []
    
#     print(f"Running {len(missing)} LSTM simulations...")
    
#     # Sequence parameters
#     n_steps = 1
    
#     for nf, layers, epochs, batch_size, lr, mod in sorted(missing):
#         subset = features[:nf]
        
#         X_tr = df_train[subset].values
#         X_te = df_test[subset].values
#         y_tr = df_train['value'].values.reshape(-1, 1)
#         y_te = df_test['value'].values.reshape(-1, 1)
        
#         # Scaling
#         scaler_X = MinMaxScaler()
#         scaler_y = MinMaxScaler()
#         X_tr_scaled = scaler_X.fit_transform(X_tr)
#         X_te_scaled = scaler_X.transform(X_te)
#         y_tr_scaled = scaler_y.fit_transform(y_tr)
#         y_te_scaled = scaler_y.transform(y_te)
        
#         # Create sequences
#         X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr_scaled, n_steps)
#         X_te_seq, y_te_seq = create_sequences(X_te_scaled, y_te_scaled, n_steps)
        
#         X_tr_seq = X_tr_seq.reshape((X_tr_seq.shape[0], n_steps, nf))
#         X_te_seq = X_te_seq.reshape((X_te_seq.shape[0], n_steps, nf))

#         # --- START OF THE FIX ---
#         # 1. Define the number of threads you want to use.
#         #    Using half of the available CPU cores is a safe and common practice.
#         num_threads = mp.cpu_count() // 2

#         # 2. Configure TensorFlow's global thread pools. This must be done
#         #    before any models are built or trained.
#         tf.config.threading.set_inter_op_parallelism_threads(num_threads)
#         tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        
#         t0 = time.perf_counter()
        
#         # Build and train model
#         model = Sequential()
#         model.add(Input(shape=(n_steps, nf)))
#         for i, units in enumerate(layers):
#             return_seq = (i < len(layers) - 1)
#             model.add(LSTM(units, return_sequences=return_seq))
#         model.add(Dense(1))


#         # Convert your numpy arrays to a tf.data.Dataset
#         train_dataset = tf.data.Dataset.from_tensor_slices((X_tr_seq, y_tr_seq))
#         train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#         model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
#         model.fit(train_dataset, epochs=epochs, verbose=0)


#         # Predictions
#         y_pred = scaler_y.inverse_transform(model.predict(X_te_seq, verbose=0)).flatten()
#         y_train_pred = scaler_y.inverse_transform(model.predict(X_tr_seq, verbose=0)).flatten()


#         elapsed = time.perf_counter() - t0

#         y_te_seq_inv = scaler_y.inverse_transform(y_te_seq).flatten()
#         y_tr_seq_inv = scaler_y.inverse_transform(y_tr_seq).flatten()
        
#         # Metrics
#         test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te_seq_inv, y_pred)) / np.mean(y_te_seq_inv)
#         test_mape = 100 * np.mean(np.abs((y_te_seq_inv - y_pred) / y_te_seq_inv))
#         train_rmse_pct = 100 * np.sqrt(mean_squared_error(y_tr_seq_inv, y_train_pred)) / np.mean(y_tr_seq_inv)
#         train_mape = 100 * np.mean(np.abs((y_tr_seq_inv - y_train_pred) / y_tr_seq_inv))
        
#         results_new.append({
#             'mode': mod,
#             'n_features': nf,
#             'layers': str(list(layers)),
#             'learning_rate': lr,
#             'epochs': epochs,
#             'batch_size': batch_size,
#             'rmse_percentual_train': train_rmse_pct,
#             'mape_train': train_mape,
#             'rmse_percentual_test': test_rmse_pct,
#             'mape_test': test_mape,
#             'processing_time': elapsed
#         })
        
#         print(f"LSTM: nf={nf}, layers={layers}, epochs={epochs}, batch={batch_size}, lr={lr}, "
#               f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
#               f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
#               f"time={elapsed:.1f}s")
    
#     # Save new results
#     if results_new:
#         df_new = pd.DataFrame(results_new)
#         df_new.to_csv(results_path, mode='a', header=False, index=False)
#         print(f"✅ Saved {len(results_new)} LSTM results")

def run_lstm_simulations(results_path, df_train, df_test, features, mode):
    """Manages the simulation runs for the LSTM model."""

    # 1. Define Hyperparameters
    hp_grid = get_hyperparameter_grid('lstm') # Assumes this function exists
    layer_configs = hp_grid['layer_configurations']
    epochs_list = hp_grid['epochs']
    batch_sizes = hp_grid['batch_size']
    learning_rates = hp_grid['learning_rates']
    
    # 2. Find Missing Simulations
    df_existing = pd.read_csv(results_path)
    existing_tuples = set()
    for _, row in df_existing.iterrows():
        try:
            # Convert layer string back to tuple for accurate checking
            layers = tuple(eval(row['layers'])) if isinstance(row['layers'], str) else tuple(row['layers'])
            existing_tuples.add((
                row['n_features'], layers, row['epochs'], 
                row['batch_size'], row['learning_rate'], row['mode']
            ))
        except (TypeError, NameError, SyntaxError):
            continue

    full_tuples = {
        (nf, tuple(layers), epochs, batch_size, lr, mode)
        for nf in range(1, len(features) + 1)
        for layers in layer_configs
        for epochs in epochs_list
        for batch_size in batch_sizes
        for lr in learning_rates
    }
    
    missing = full_tuples - existing_tuples
    if not missing:
        print("✅ No new LSTM simulations to run.")
        return

    print(f"Running {len(missing)} new LSTM simulations...")
    
    # 2. Run simulations and save each result immediately
    for nf, layers, epochs, batch_size, lr, mod in sorted(missing):
        forecast_args = {
            'df_train': df_train,
            'df_test': df_test,
            'features': features,
            'mod': mod,
            'nf': nf,
            'layers': layers,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr': lr
        }
        
        result, y_pred_test = model_forecast("LSTM", **forecast_args)

        # --- START OF THE FIX ---
        # Convert the single result dictionary to a DataFrame
        df_new = pd.DataFrame([result])
        
        # Check if the file needs a header
        header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0
        
        # Append the single result to the CSV file
        df_new.to_csv(results_path, mode='a', header=header, index=False)
        # --- END OF THE FIX ---


def random_forest_forecast(df_train, df_test, features, mod, nf, depth, ne, **kwargs) -> dict:
    """Trains and evaluates a single Random Forest model."""
    subset = features[:nf]
    X_sub_tr, y_tr = df_train[subset].values, df_train['value'].values.ravel()
    X_sub_te, y_te = df_test[subset].values, df_test['value'].values.ravel()
    
    t0 = time.perf_counter()
    model = RandomForestRegressor(
        max_depth=depth,
        n_estimators=ne,
        n_jobs=mp.cpu_count() // 2,
        random_state=42,
        verbose=0
    )
    model.fit(X_sub_tr, y_tr)
    
    # Predictions & Metrics
    y_pred_test = model.predict(X_sub_te)
    elapsed = time.perf_counter() - t0
    y_pred_train = model.predict(X_sub_tr)
    
    train_rmse_pct = 100 * np.sqrt(mean_squared_error(y_tr, y_pred_train)) / np.mean(y_tr)
    train_mape = 100 * np.mean(np.abs((y_tr - y_pred_train) / y_tr))
    test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te, y_pred_test)) / np.mean(y_te)
    test_mape = 100 * np.mean(np.abs((y_te - y_pred_test) / y_te))
    
    print(f"RF: nf={nf}, depth={depth}, ne={ne}, "
            f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
            f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
            f"time={elapsed:.1f}s")
    
    return {
        'mode': mod, 'n_features': nf, 'max_depth': depth, 'n_estimators': ne,
        'rmse_percentual_train': train_rmse_pct, 'mape_train': train_mape,
        'rmse_percentual_test': test_rmse_pct, 'mape_test': test_mape,
        'processing_time': elapsed
    }, y_pred_test

def xgboost_forecast(df_train, df_test, features, mod, nf, depth, ne, lr, **kwargs) -> dict:
    """Trains and evaluates a single XGBoost model."""
    subset = features[:nf]
    X_sub_tr, y_tr = df_train[subset].values, df_train['value'].values.ravel()
    X_sub_te, y_te = df_test[subset].values, df_test['value'].values.ravel()
    
    t0 = time.perf_counter()
    model = xgb.XGBRegressor(
        max_depth=depth,
        n_estimators=int(ne),
        learning_rate=lr,
        n_jobs=mp.cpu_count() // 2,
        tree_method="hist",
        verbosity=0,
        random_state=42
    )
    model.fit(X_sub_tr, y_tr)
    elapsed = time.perf_counter() - t0
    
    # Predictions & Metrics
    y_pred_train = model.predict(X_sub_tr)
    y_pred_test = model.predict(X_sub_te)
    train_rmse_pct = 100 * np.sqrt(mean_squared_error(y_tr, y_pred_train)) / np.mean(y_tr)
    train_mape = 100 * np.mean(np.abs((y_tr - y_pred_train) / y_tr))
    test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te, y_pred_test)) / np.mean(y_te)
    test_mape = 100 * np.mean(np.abs((y_te - y_pred_test) / y_te))

    print(f"XGB: nf={nf}, depth={depth}, ne={ne}, lr={lr}, "
            f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
            f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
            f"time={elapsed:.1f}s")
    
    return {
        'mode': mod, 'n_features': nf, 'max_depth': depth, 'learning_rate': lr,
        'n_estimators': ne, 'rmse_percentual_train': train_rmse_pct,
        'mape_train': train_mape, 'rmse_percentual_test': test_rmse_pct,
        'mape_test': test_mape, 'processing_time': elapsed
    }, y_pred_test

def lstm_forecast(df_train, df_test, features, mod, nf, layers, epochs, batch_size, lr, **kwargs) -> dict:
    """Trains and evaluates a single LSTM model."""
    # Assumes `create_sequences` function is defined elsewhere
    subset = features[:nf]
    X_tr, y_tr = df_train[subset].values, df_train['value'].values.reshape(-1, 1)
    X_te, y_te = df_test[subset].values, df_test['value'].values.reshape(-1, 1)

    # Scaling
    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_tr_scaled, y_tr_scaled = scaler_X.fit_transform(X_tr), scaler_y.fit_transform(y_tr)
    X_te_scaled, y_te_scaled = scaler_X.transform(X_te), scaler_y.transform(y_te)
    
    # Create sequences
    n_steps = 1 # Assuming a look-back of 1
    X_tr_seq, y_tr_seq = create_sequences(X_tr_scaled, y_tr_scaled, n_steps)
    X_te_seq, y_te_seq = create_sequences(X_te_scaled, y_te_scaled, n_steps)

    num_threads = mp.cpu_count() // 2

    # 2. Configure TensorFlow's global thread pools. This must be done
    #    before any models are built or trained.
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    
    t0 = time.perf_counter()
    model = Sequential([Input(shape=(n_steps, nf))])
    for i, units in enumerate(layers):
        model.add(LSTM(units, return_sequences=(i < len(layers) - 1)))
    model.add(Dense(1))
    
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    model.fit(X_tr_seq, y_tr_seq, epochs=epochs, batch_size=batch_size, verbose=0)
    elapsed = time.perf_counter() - t0
    
    # Predictions & Metrics
    y_pred_train_scaled = model.predict(X_tr_seq, verbose=0)
    y_pred_test_scaled = model.predict(X_te_seq, verbose=0)
    y_train_pred = scaler_y.inverse_transform(y_pred_train_scaled).flatten()
    y_pred = scaler_y.inverse_transform(y_pred_test_scaled).flatten()
    y_tr_seq_inv = scaler_y.inverse_transform(y_tr_seq).flatten()
    y_te_seq_inv = scaler_y.inverse_transform(y_te_seq).flatten()
    
    train_rmse_pct = 100*np.sqrt(mean_squared_error(y_tr_seq_inv, y_train_pred))/np.mean(y_tr_seq_inv)
    train_mape = 100 * np.mean(np.abs((y_tr_seq_inv - y_train_pred) / y_tr_seq_inv))
    test_rmse_pct = 100 * np.sqrt(mean_squared_error(y_te_seq_inv, y_pred)) / np.mean(y_te_seq_inv)
    test_mape = 100 * np.mean(np.abs((y_te_seq_inv - y_pred) / y_te_seq_inv))

    print(f"LSTM: nf={nf}, layers={layers}, epochs={epochs}, batch={batch_size}, lr={lr}, "
              f"Train RMSE% = {train_rmse_pct:.2f}%, Test RMSE% = {test_rmse_pct:.2f}%, "
              f"Train MAPE = {train_mape:.2f}%, Test MAPE = {test_mape:.2f}%, "
              f"time={elapsed:.1f}s")
    
    return {
        'mode': mod, 'n_features': nf, 'layers': str(list(layers)),
        'learning_rate': lr, 'epochs': epochs, 'batch_size': batch_size,
        'rmse_percentual_train': train_rmse_pct, 'mape_train': train_mape,
        'rmse_percentual_test': test_rmse_pct, 'mape_test': test_mape,
        'processing_time': elapsed
    }, y_pred


def create_sequences(X, y, n_steps):
    """Create sequences for LSTM input."""
    X_seq, y_seq = [], []
    for i in range(len(X) - n_steps):
        X_seq.append(X[i:i + n_steps])
        y_seq.append(y[i + n_steps])
    return np.array(X_seq), np.array(y_seq)


def plot_tree_model_results(results_path: str, model_name: str, country: str, nationality: str, time_limit: float):
    """
    Reads and plots results for tree-based models (Random Forest, XGBoost).
    """
    # Define the hyperparameter pairs to plot for each model
    hp_map = {
        "Random_Forest": [
            ("n_estimators", "n_features"),
            ("max_depth", "n_features"),
        ],
        "XGBoost": [
            ("n_estimators", "n_features"),
            ("max_depth", "n_features"),
            ("learning_rate", "n_features"),
        ]
    }
    
    # Read and prepare the dataframe
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return
        
    df.columns = df.columns.str.strip()

    # Separate data for different modes
    df_fr = df[df["mode"] == "Fourier Ramps"].copy()
    df_cal = df[df["mode"] == "Calendar"].copy()

    # --- START OF THE SOLUTION ---

    # First, make sure you know the exact name of your RMSE column.
    # Based on our previous work, it's likely 'rmse_percentual_test'.
    rmse_column_name = 'rmse_percentual_test'

    # Find the row with the lowest RMSE in the entire DataFrame 🏆
    best_overall_idx = df[rmse_column_name].idxmin()
    best_overall_row = df.loc[best_overall_idx]

    print("--- Best Overall Result ---")
    print(best_overall_row)
    print("\n" + "="*30 + "\n")


    # Find the best result specifically for the 'Fourier Ramps' mode
    if not df_fr.empty:
        best_fr_idx = df_fr[rmse_column_name].idxmin()
        best_fr_row = df_fr.loc[best_fr_idx]
        print("--- Best Fourier Ramps Result ---")
        print(best_fr_row)
        print("\n" + "="*30 + "\n")


    # Find the best result specifically for the 'Calendar' mode
    if not df_cal.empty:
        best_cal_idx = df_cal[rmse_column_name].idxmin()
        best_cal_row = df_cal.loc[best_cal_idx]
        print("--- Best Calendar Result ---")
        print(best_cal_row)
        
    # --- END OF THE SOLUTION ---

    def make_figure(xaxis: str, yaxis: str, metric: str, zlabel: str):
        fig = go.Figure()
        metric_train = f"{metric.lower()}_train"
        metric_test = f"{metric.lower()}_test"

        # 1. Fourier Ramps (Train)
        df_fr_train_pivot = df_fr.pivot_table(index=xaxis, columns=yaxis, values=metric_train, aggfunc="min")
        fig.add_trace(go.Surface(x=df_fr_train_pivot.columns, y=df_fr_train_pivot.index, z=df_fr_train_pivot.values, name='Fourier Ramps (Train)', colorscale='Blues', opacity=0.7, showscale=False, showlegend=True))

        # 2. Fourier Ramps (Test)
        df_fr_test_pivot = df_fr.pivot_table(index=xaxis, columns=yaxis, values=metric_test, aggfunc="min")
        fig.add_trace(go.Surface(x=df_fr_test_pivot.columns, y=df_fr_test_pivot.index, z=df_fr_test_pivot.values, name='Fourier Ramps (Test)', colorscale='Viridis', opacity=0.7, showscale=False, showlegend=True))

        # Filter for the slow simulations that lie on the surface

        # 1. For each (xaxis, yaxis) pair in the original df_fr, find the index
        #    of the row that has the minimum error. This gives us the exact
        #    set of data points that are used to draw the surface.
        idx_of_surface_points = df_fr.groupby([xaxis, yaxis])[metric_test].idxmin()

        # 2. Create a new DataFrame containing only these best-performing rows.
        #    This DataFrame now represents every point on your surface and
        #    still has the 'processing_time' column.
        df_surface_points = df_fr.loc[idx_of_surface_points]

        # 3. From this new DataFrame of surface points, filter for the ones
        #    that were slow. This is the final, precise set of markers to plot.
        df_markers = df_surface_points[df_surface_points['processing_time'] > time_limit]


        if time_limit < 60:
            time_message = f"{time_limit} seconds"
        else:
            time_message = f"{time_limit / 60:.1f} minutes"

        # Add the Scatter3d trace using the new, perfectly aligned marker data.
        if not df_markers.empty:
            fig.add_trace(go.Scatter3d(
                x=df_markers[yaxis],
                y=df_markers[xaxis],
                z=df_markers[metric_test],  # The Z-value is guaranteed to be on the surface
                mode='markers',
                showlegend=True,
                name=f'Slow > {time_message}', # Still useful for hover info
                marker=dict(
                    color='yellow',
                    symbol='diamond',
                    size=5,
                    line=dict(color='black', width=1)
                ),
                hovertemplate = 
                    '<b>Slow Simulation</b><br>' +
                    f'{xaxis}: ' + '%{y}<br>' +
                    f'{yaxis}: ' + '%{x}<br>' +
                    f'{metric}: ' + '%{z:.2f}<br>' +
                    'Time: %{text:.2f}s' +
                    '<extra></extra>',
                text = df_markers['processing_time']
            ))

        # 3. Calendar (Test)
        df_cal_pivot = df_cal.pivot_table(index=xaxis, columns=yaxis, values=metric_test, aggfunc="min")
        fig.add_trace(go.Surface(x=df_cal_pivot.columns, y=df_cal_pivot.index, z=df_cal_pivot.values, name='Calendar (Test)', colorscale='YlOrBr', opacity=0.9, showscale=False, showlegend=True))

        scene_config = {
            'xaxis_title': yaxis.replace('_', ' ').title(),
            'yaxis_title': xaxis.replace('_', ' ').title(),
            'zaxis_title': zlabel,
            'aspectmode': "cube"
        }

        # 2. Dynamically add the log scale setting if 'learning_rate' is one of the axes.
        #    Note: In Plotly's 3D scene, the `yaxis` variable from your function corresponds to the `xaxis` of the plot,
        #    and your `xaxis` variable corresponds to the `yaxis` of the plot.
        if yaxis == 'learning_rate':
            scene_config['xaxis'] = {'type': 'log'}
        
        if xaxis == 'learning_rate':
            scene_config['yaxis'] = {'type': 'log'}

        # 3. Update the figure layout with the dynamically created scene configuration
        fig.update_layout(
            title_text=f"{model_name} {metric} ({xaxis.replace('_', ' ')} vs {yaxis.replace('_', ' ')}) for {nationality} dataset",
            height=700,
            width=900,
            legend_title_text='Results',
            legend=dict(
                font=dict(size=18),
                x=0.1,  # horizontal position (0-1)
                y=1,  # vertical position (0-1)
                bgcolor="rgba(255,255,255,0.8)",  # semi-transparent white background
                bordercolor="white",
                borderwidth=1
            ),
            scene=scene_config,  # Use the dynamic config dictionary here
            margin=dict(l=0, r=150, b=0, t=50)
        )
        

        fig.show()


    # Generate figures for all hyperparameter pairs
    for i, (hp1, hp2) in enumerate(hp_map[model_name]):
        print(f"--- Generating {model_name} figures for {country} ({hp1} vs {hp2}) ---")
        make_figure(hp1, hp2, "RMSE_percentual", "RMSE (%)")
        make_figure(hp1, hp2, "MAPE", "MAPE (%)")


def plot_lstm_results(results_path: str, model_name: str, country: str, nationality: str, time_limit: float):
    """
    Reads, cleans, and plots LSTM simulation results from a CSV file, creating
    separate 3D surface plots for different hyperparameter comparisons.
    """
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_path}' was not found.")
        return
        
    # --- 1. Initial Data Cleaning and Preparation ---
    df.columns = df.columns.str.strip()
    df['layers'] = df['layers'].apply(ast.literal_eval).apply(tuple)

    # Clean data by keeping only the single best result for each unique hyperparameter set
    hyperparameter_cols = ['mode', 'n_features', 'layers', 'learning_rate', 'epochs', 'batch_size']
    df = df.sort_values(by='rmse_percentual_test', ascending=True)
    df = df.drop_duplicates(subset=hyperparameter_cols, keep='first')

    # --- 2. Find and Print the Best Overall Result ---
    rmse_column_name = 'rmse_percentual_test'
    if rmse_column_name in df.columns:
        best_overall_idx = df[rmse_column_name].idxmin()
        best_overall_row = df.loc[best_overall_idx]
        print("--- 🏆 Best Overall Result ---")
        print(best_overall_row)
        print("\n" + "="*40 + "\n")

    # Separate data for different modes
    df_fr = df[df["mode"] == "Fourier Ramps"].copy()
    df_cal = df[df["mode"] == "Calendar"].copy()

    # Find the best result specifically for the 'Fourier Ramps' mode
    if not df_fr.empty:
        best_fr_idx = df_fr[rmse_column_name].idxmin()
        best_fr_row = df_fr.loc[best_fr_idx]
        print("--- Best Fourier Ramps Result ---")
        print(best_fr_row)
        print("\n" + "="*40 + "\n")

    # Find the best result specifically for the 'Calendar' mode
    if not df_cal.empty:
        best_cal_idx = df_cal[rmse_column_name].idxmin()
        best_cal_row = df_cal.loc[best_cal_idx]
        print("--- Best Calendar Result ---")
        print(best_cal_row)
        print("\n" + "="*40 + "\n")
    
    # --- 3. Helper Function to Create Each Figure ---
    def make_figure(df_full, xaxis_name, yaxis_name, metric, zlabel):
        fig = go.Figure()
        metric_train_col = f"{metric.lower()}_train"
        metric_test_col = f"{metric.lower()}_test"

        df_fr = df_full[df_full['mode'] == 'Fourier Ramps'].copy()
        df_cal = df_full[df_full['mode'] == 'Calendar'].copy()

        y_axis_config = {}
        pivot_index_col = xaxis_name
        
        # --- FIX: Use pd.Categorical for robust 'layers' axis handling ---
        if xaxis_name == 'layers':
            # Define the complete, sorted order for the categorical axis
            all_layers_sorted = sorted(df_full['layers'].unique(), key=lambda x: (len(x), x))
            
            # Convert the 'layers' column to a categorical type with the specified order
            if not df_fr.empty:
                df_fr['layers'] = pd.Categorical(df_fr['layers'], categories=all_layers_sorted, ordered=True)
                df_fr['layers_numeric'] = df_fr['layers'].cat.codes
            if not df_cal.empty:
                df_cal['layers'] = pd.Categorical(df_cal['layers'], categories=all_layers_sorted, ordered=True)
                df_cal['layers_numeric'] = df_cal['layers'].cat.codes

            pivot_index_col = 'layers_numeric'
            y_axis_config = dict(
                tickmode='array',
                tickvals=list(range(len(all_layers_sorted))),
                ticktext=[str(l) for l in all_layers_sorted]
            )

        # --- Plot Surfaces and Markers ---
        if not df_fr.empty:
            pivot_train = df_fr.pivot_table(index=pivot_index_col, columns=yaxis_name, values=metric_train_col, aggfunc='min')
            fig.add_trace(go.Surface(x=pivot_train.columns, y=pivot_train.index, z=pivot_train.values, name='Fourier Ramps (Train)', colorscale='Blues', opacity=0.7, showscale=False, showlegend=True))
            pivot_test = df_fr.pivot_table(index=pivot_index_col, columns=yaxis_name, values=metric_test_col, aggfunc='min')
            fig.add_trace(go.Surface(x=pivot_test.columns, y=pivot_test.index, z=pivot_test.values, name='Fourier Ramps (Test)', colorscale='Viridis', opacity=0.7, showscale=False, showlegend=True))

            # Markers for Slow Simulations
            idx_surface = df_fr.groupby([xaxis_name, yaxis_name], observed=True)[metric_test_col].idxmin()
            df_markers = df_fr.loc[idx_surface][df_fr.loc[idx_surface]['processing_time'] > time_limit]
            if not df_markers.empty:
                marker_y = df_markers[xaxis_name]
                if xaxis_name == 'layers':
                    marker_y = df_markers['layers'].cat.codes
                
                fig.add_trace(go.Scatter3d(
                    x=df_markers[yaxis_name], y=marker_y, z=df_markers[metric_test_col],
                    mode='markers', showlegend=True, name=f'Slow (> {time_limit}s)',
                    marker=dict(color='gold', symbol='diamond', size=5, line=dict(color='black', width=1)),
                    hovertemplate = f'<b>Slow Sim</b><br>{xaxis_name}: %{{customdata[0]}}<br>{yaxis_name}: %{{x}}<br>{metric}: %{{z:.2f}}<br>Time: %{{text:.2f}}s<extra></extra>',
                    text=df_markers['processing_time'], customdata=df_markers[[xaxis_name]]
                ))

        if not df_cal.empty:
            pivot_cal = df_cal.pivot_table(index=pivot_index_col, columns=yaxis_name, values=metric_test_col, aggfunc='min')
            fig.add_trace(go.Surface(x=pivot_cal.columns, y=pivot_cal.index, z=pivot_cal.values, name='Calendar (Test)', colorscale='YlOrBr', opacity=0.9, showscale=False, showlegend=True))
        
        # --- Finalize Layout ---
        scene_config = {
            'xaxis_title': yaxis_name.replace('_', ' ').title(),
            'yaxis_title': xaxis_name.replace('_', ' ').title(),
            'zaxis_title': zlabel, 'aspectmode': "cube"
        }
        if yaxis_name == 'learning_rate': scene_config['xaxis'] = {'type': 'log'}
        if xaxis_name == 'learning_rate': scene_config['yaxis'] = {'type': 'log'}

        fig.update_layout(
            title_text=f"{model_name} {metric} ({xaxis_name.replace('_', ' ')} vs {yaxis_name.replace('_', ' ')})",
            scene=scene_config, legend_title_text='Results',
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=20)),
            legend=dict(
                font=dict(size=18),
                x=0.1,  # horizontal position (0-1)
                y=1,  # vertical position (0-1)
                bgcolor="rgba(255,255,255,0.8)",  # semi-transparent white background
                bordercolor="white",
                borderwidth=1
            ),
            height=700, width=900, margin=dict(l=0, r=0, b=0, t=50)
        )
        
        if y_axis_config:
            fig.update_scenes(yaxis=y_axis_config)

        fig.show()

    # --- 4. Define Hyperparameter Pairs and Generate Plots ---
    hp_pairs = [
        ("layers", "n_features"),
        ("batch_size", "n_features"),
        ("epochs", "n_features"),
        ("learning_rate", "n_features")
    ]
    
    for hp1, hp2 in hp_pairs:
        print(f"--- Generating {model_name} figures for {country} ({hp1} vs {hp2}) ---")
        make_figure(df, hp1, hp2, "RMSE_percentual", "RMSE (%)")
        make_figure(df, hp1, hp2, "MAPE", "MAPE (%)")

# def plot_lstm_results(results_path: str, model_name: str, country: str, nationality: str, time_limit: float):
#     """
#     Reads and plots results for LSTM models, pivoting data around the best result.
#     """
#     try:
#         df = pd.read_csv(results_path)
#     except FileNotFoundError:
#         print(f"Error: The file '{results_path}' was not found.")
#         return
        
#     df.columns = df.columns.str.strip()
#     df['layers'] = df['layers'].apply(ast.literal_eval).apply(tuple)

#     # Clean the data by keeping only the best result for each unique set of HPs
#     hyperparameter_cols = ['mode', 'n_features', 'layers', 'learning_rate', 'epochs', 'batch_size']
#     df = df.sort_values(by='rmse_percentual_test', ascending=True)
#     df = df.drop_duplicates(subset=hyperparameter_cols, keep='first')

    
#     df_fr = df[df["mode"] == "Fourier Ramps"].copy()
#     df_cal = df[df["mode"] == "Calendar"].copy()


#     # First, make sure you know the exact name of your RMSE column.
#     # Based on our previous work, it's likely 'rmse_percentual_test'.
#     rmse_column_name = 'rmse_percentual_test'

#     # Find the row with the lowest RMSE in the entire DataFrame 🏆
#     best_overall_idx = df[rmse_column_name].idxmin()
#     best_overall_row = df.loc[best_overall_idx]

#     print("--- Best Overall Result ---")
#     print(best_overall_row)
#     print("\n" + "="*30 + "\n")


#     # Find the best result specifically for the 'Fourier Ramps' mode
#     if not df_fr.empty:
#         best_fr_idx = df_fr[rmse_column_name].idxmin()
#         best_fr_row = df_fr.loc[best_fr_idx]
#         print("--- Best Fourier Ramps Result ---")
#         print(best_fr_row)
#         print("\n" + "="*30 + "\n")


#     # Find the best result specifically for the 'Calendar' mode
#     if not df_cal.empty:
#         best_cal_idx = df_cal[rmse_column_name].idxmin()
#         best_cal_row = df_cal.loc[best_cal_idx]
#         print("--- Best Calendar Result ---")
#         print(best_cal_row)
        
#     # --- END OF THE SOLUTION ---

#     def make_figure_from_pivot(df_sub, xaxis_name, yaxis_name, metric, zlabel):
#         fig = go.Figure()
#         metric_train_col = f"{metric.lower()}_train"
#         metric_test_col = f"{metric.lower()}_test"

#         df_fr_all = df_sub[df_sub['mode'] == 'Fourier Ramps'].copy()
#         df_cal_all = df_sub[df_sub['mode'] == 'Calendar'].copy()

#         df_fr, df_cal = pd.DataFrame(), pd.DataFrame() # Initialize empty dataframes

#         # --- 2. Process Fourier Ramps Data ---
#         if not df_fr_all.empty:
#             best_row_fr = df_fr_all.loc[df_fr_all[metric_test_col].idxmin()]
#             constant_hps = {'layers', 'epochs', 'batch_size', 'learning_rate'} - {xaxis_name, yaxis_name}
#             mask_fr = pd.Series(True, index=df_fr_all.index)
#             for hp in constant_hps:
#                 mask_fr &= (df_fr_all[hp] == best_row_fr[hp])
#             df_fr = df_fr_all[mask_fr].copy()

#         # --- 3. Process Calendar Data ---
#         if not df_cal_all.empty:
#             best_row_cal = df_cal_all.loc[df_cal_all[metric_test_col].idxmin()]
#             constant_hps = {'layers', 'epochs', 'batch_size', 'learning_rate'} - {xaxis_name, yaxis_name}
#             mask_cal = pd.Series(True, index=df_cal_all.index)
#             for hp in constant_hps:
#                 mask_cal &= (df_cal_all[hp] == best_row_cal[hp])
#             df_cal = df_cal_all[mask_cal].copy()

#         # --- 4. Configure Categorical 'layers' Axis (The Fix) ---
#         y_axis_config = {}
#         pivot_index_col = xaxis_name

#         if xaxis_name == 'layers':
#             # Combine the final filtered dataframes to get all unique layers for the axis
#             combined_filtered = pd.concat([df_fr, df_cal])
            
#             if not combined_filtered.empty:
#                 sorted_uniques = sorted(combined_filtered['layers'].unique(), key=lambda x: (len(x), x))
                
#                 tick_values = list(range(len(sorted_uniques)))
#                 tick_labels = [str(l) for l in sorted_uniques]
                
#                 layers_map = {layer: i for i, layer in enumerate(sorted_uniques)}
                
#                 # Apply the numeric mapping to both dataframes
#                 if not df_fr.empty:
#                     df_fr['layers_numeric'] = df_fr['layers'].map(layers_map)
#                 if not df_cal.empty:
#                     df_cal['layers_numeric'] = df_cal['layers'].map(layers_map)
                
#                 pivot_index_col = 'layers_numeric'
#                 y_axis_config = dict(tickmode='array', tickvals=tick_values, ticktext=tick_labels)


#         # 2. Process and plot Fourier Ramps if data exists for it.
#         if not df_fr.empty:
#             # Find the best row *within the Fourier Ramps data*
#             best_row_fr = df_fr.loc[df_fr[metric_test_col].idxmin()]

#             # Filter the Fourier data based on its own best hyperparameters
#             constant_hps = {'layers', 'epochs', 'batch_size', 'learning_rate'} - {xaxis_name, yaxis_name}
#             mask_fr = pd.Series(True, index=df_fr.index)
#             for hp in constant_hps:
#                 mask_fr &= (df_fr[hp] == best_row_fr[hp])
#             df_fr = df_fr[mask_fr].copy() # This is the final data for the FR plot

#             # Plotting logic for Fourier Ramps surfaces
#             pivot_fr_train = df_fr.pivot_table(index=xaxis_name, columns=yaxis_name, values=metric_train_col, aggfunc='min')
#             fig.add_trace(go.Surface(x=pivot_fr_train.columns, y=pivot_fr_train.index, z=pivot_fr_train.values, name='Fourier Ramps (Train)', colorscale='Blues', opacity=0.7, showscale=False, showlegend=True))
            
#             pivot_fr_test = df_fr.pivot_table(index=xaxis_name, columns=yaxis_name, values=metric_test_col, aggfunc='min')
#             fig.add_trace(go.Surface(x=pivot_fr_test.columns, y=pivot_fr_test.index, z=pivot_fr_test.values, name='Fourier Ramps (Test)', colorscale='Viridis', opacity=0.7, showscale=False, showlegend=True))
            
#             # Marker plotting logic for slow simulations (now correctly uses df_fr)
#             idx_of_surface_points = df_fr.groupby([xaxis_name, yaxis_name])[metric_test_col].idxmin()
#             df_surface_points = df_fr.loc[idx_of_surface_points]
#             df_markers = df_surface_points[df_surface_points['processing_time'] > time_limit]
            
            
#             if time_limit < 60:
#                 time_message = f"{time_limit} seconds"
#             else:
#                 time_message = f"{time_limit / 60:.1f} minutes"

#             if not df_markers.empty:
#                 fig.add_trace(go.Scatter3d(
#                     # 2. Use the correct variable names for x and y axes
#                     x=df_markers[yaxis_name],
#                     y=df_markers[xaxis_name],
#                     z=df_markers[metric_test_col],
#                     mode='markers',
#                     showlegend=True,
#                     name=f'Train and test time higher than {time_message}',
#                     marker=dict(
#                         color='yellow', symbol='diamond', size=5,
#                         line=dict(color='black', width=1)
#                     ),
#                     # 3. Use the correct variable names in the hovertemplate f-strings
#                     hovertemplate = 
#                         '<b>Slow Simulation</b><br>' +
#                         f'{xaxis_name}: ' + '%{y}<br>' +
#                         f'{yaxis_name}: ' + '%{x}<br>' +
#                         f'{metric}: ' + '%{z:.2f}<br>' +
#                         'Time: %{text:.2f}s' +
#                         '<extra></extra>',
#                     text = df_markers['processing_time']
#                 ))

#         if not df_cal_all.empty:
#             # Find the best row *within the Calendar data*
#             best_row_cal = df_cal_all.loc[df_cal_all[metric_test_col].idxmin()]

#             # Filter the Calendar data based on its own best hyperparameters
#             constant_hps = {'layers', 'epochs', 'batch_size', 'learning_rate'} - {xaxis_name, yaxis_name}
#             mask_cal = pd.Series(True, index=df_cal_all.index)
#             for hp in constant_hps:
#                 mask_cal &= (df_cal_all[hp] == best_row_cal[hp])
#             df_cal = df_cal_all[mask_cal].copy() # This is the final data for the Calendar plot
            
#             # Plotting logic for Calendar surface
#             pivot_cal = df_cal.pivot_table(index=xaxis_name, columns=yaxis_name, values=metric_test_col, aggfunc='min')
#             fig.add_trace(go.Surface(x=pivot_cal.columns, y=pivot_cal.index, z=pivot_cal.values, name='Calendar (Test)', colorscale='YlOrBr', opacity=0.9, showscale=False, showlegend=True))
        
#        # 1. Start with the basic scene configuration including titles
#         scene_config = {
#             'xaxis_title': yaxis_name.replace('_', ' ').title(),
#             'yaxis_title': xaxis_name.replace('_', ' ').title(),
#             'zaxis_title': zlabel,
#             'aspectmode': "cube"
#         }

#         # 2. Dynamically add the log scale setting using the correct keys: 'xaxis' and 'yaxis'
#         if yaxis_name == 'learning_rate':
#             scene_config['xaxis'] = {'type': 'log'}
        
#         if xaxis_name == 'learning_rate':
#             scene_config['yaxis'] = {'type': 'log'}

#         # 3. Update the figure layout with the dynamically created scene configuration
#         fig.update_layout(
#             title_text=f"{model_name} {metric} ({xaxis_name.replace('_', ' ')} vs {yaxis_name.replace('_', ' ')}) for {nationality} dataset",
#             height=700,
#             width=900,
#             legend_title_text='Results',
#             scene=scene_config,  # Use the dynamic config dictionary here
#             margin=dict(l=0, r=0, b=0, t=50)
#         )
        
#         if y_axis_config:
#             fig.update_scenes(yaxis=y_axis_config)

#         fig.show()

#     hp_pairs = [
#         ("layers", "n_features"), ("epochs", "n_features"), 
#         ("learning_rate", "n_features"), ("batch_size", "n_features")
#     ]
    
#     for hp1, hp2 in hp_pairs:
#         print(f"--- Generating {model_name} figures for {country} ({hp1} vs {hp2}) ---")
#         make_figure_from_pivot(df, hp1, hp2, "RMSE_percentual", "RMSE (%)")
#         make_figure_from_pivot(df, hp1, hp2, "MAPE", "MAPE (%)")



def main():
    """Example usage of the many_times module."""
    print("🚀 Many Times - ML Model Simulations")
    print("=" * 50)
    
    # Example usage
    country = "Brazil"
    model = "Random_Forest"
    mode = "calendar"
    
    print(f"Country: {country}")
    print(f"Model: {model}")
    print(f"Mode: {mode}")
    
    # Verify and create files
    try:
        results_path, plot_dir, csv_exists = verify_file(country, model)
        print(f"Results path: {results_path}")
        print(f"Plot directory: {plot_dir}")
        print(f"CSV exists: {csv_exists}")
        
        # Note: df_train and df_test would need to be loaded from your data
        # This is just a demonstration of the structure
        print("\n✅ File verification completed!")
        print("📝 Use several_simulations() to run the actual simulations")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
