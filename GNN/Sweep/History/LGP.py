import wandb

# Example sweep configuration
sweep_configuration = {
    "method": "grid",
    "name": "History-GCN-LGP",
    "metric": {"goal": "maximize", "name": "Mean_Val_accuracy"},
    "parameters": {
        "dropout": {"values": [0.2, 0.5]},
        "early_stop_patience": {"value": 50},
        "LLM_feature": {"values": ["/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/History_Llama_2_13b_hf_256_mean.npy"]},
        "PLM_feature": {"values": ["/dataintent/local/user/v-yinju/haoyan/Data/Books/History/Feature/History_roberta_base_512_Tuned_cls.npy"]},
        "graph_path": {"value": "/dataintent/local/user/v-yinju/haoyan/Data/Books/History/HistoryGraph.pt"},
        "label-smoothing": {"values": [0.1, 0.3]},
        "lr": {"values": [0.0005, 0.001, 0.005]},
        "n-hidden": {"values": [256, 128]},
        "n-layers": {"values": [2, 3]},
        "n-runs": {"value": 5},
        "warmup_epochs": {"value": 50}
    },
    "program": "GNN/Library/LPG.py",
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="History")
