import flwr as fl
from flwr.server.strategy import FedAvg
import mlflow

def weighted_average(metrics):
    """Aggregate AUC scores from all clients"""
    if not metrics:
        return {}
    
    # metrics is a list of (num_examples, metrics_dict)
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_auc = sum(num_examples * m.get("auc", 0) for num_examples, m in metrics) / total_examples
    
    print(f"📊 Aggregated AUC: {weighted_auc:.3f}")
    mlflow.log_metric("federated_auc", weighted_auc)
    
    return {"auc": weighted_auc}

def start_server(num_rounds=3, min_clients=2, port=5050):
    """Launch federated training"""
    mlflow.set_experiment("federated_health_risk")
    
    strategy = FedAvg(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_metrics_aggregation_fn=weighted_average,  # Add this
    )
    
    fl.server.start_server(
        server_address=f"0.0.0.0:{port}",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )