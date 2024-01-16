import wandb

# Example sweep configuration
sweep_configuration = {
    "method": "grid",
    "name": "Children-RevGAT-LGP",
    "metric": {"goal": "maximize", "name": "Mean_Val_accuracy"},
    "parameters": {
        "LLM_feature": {"values": ["/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/Children_Llama_2_13b_hf_256_mean.npy"]},
        "PLM_feature": {"values": ["/dataintent/local/user/v-yinju/haoyan/Data/Books/Children/Feature/Children_roberta_base_512_Tuned_cls.npy"]},
        "dropout": {"values": [0.5]},
        "edge-drop": {"value": 0.3},
        "attn-drop": {"value": 0.0},
        "early_stop_patience": {"value": 50},
        "alpha": {"values": [0.5, 0.7, 0.9]},
        "conv_type": {"values": ["SAGE", "Linear"]},
        "gnn": {"value": "RevGAT"},
        "label-smoothing": {"values": [0.3]},
        "lr": {"values": [0.0005, 0.001, 0.005, 0.01]},
        "n-hidden": {"values": [256, 128]},
        "n-layers": {"values": [5]},
        "n-runs": {"value": 5},
        "warmup_epochs": {"value": 1},
    },
    "program": "GNN/Library/LPG.py",
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Children")
