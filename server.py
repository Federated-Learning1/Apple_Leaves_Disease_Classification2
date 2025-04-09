# There is one server and 3 clients
# server.py
# Use the flower server 
import flwr as fl

# Optional: Configure training rounds (e.g., if you want to send hyperparameters)
def fit_config(rnd: int):
    return {
        "round": rnd  # This will be available in client fit()
    }

# This is used to pass the round number into the client's evaluate() method
def evaluate_config(rnd: int):
    return {
        "round": rnd  # So client can save model on final round
    }

# Define the federated averaging strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_available_clients=3,
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=evaluate_config,
)

# Start the Flower server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=5),  # Number of federated training rounds
    strategy=strategy,
)
