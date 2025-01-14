import wandb

# 配置sweep参数
sweep_config = {
    "method": "grid",
    "metric": {
        "goal": "maximize",
        "name": "Mean_Test_accuracy"
    },
    "name": "Reddit-GAT",
    "parameters": {
        "attn-drop": {
            "values": [0]
        },
        "average": {
            "value": "macro"
        },
        "dropout": {
            "values": [0.5, 0.75]
        },
        "early_stop_patience": {
            "value": 50
        },
        "edge-drop": {
            "values": [0.25]
        },
        "feature": {
            "values": [
                "Data/RedditS/TextFeature/RedditS_ModernBERT_base_100_mean.npy",
                "Data/RedditS/TextFeature/RedditS_roberta_base_100_mean.npy",
                "Data/RedditS/TextFeature/RedditS_Llama_3.2_1B_Instruct_100_mean.npy",
                "Data/RedditS/TextFeature/RedditS_Llama_3.2_3B_Instruct_100_mean.npy",
                "Data/RedditS/TextFeature/RedditS_Llama_3.1_8B_Instruct_100_mean.npy",
                "Data/RedditS/TextFeature/RedditS_Ministral_8B_Instruct_2410_100_mean.npy"
            ]
        },
        "graph_path": {
            "value": "Data/RedditS/RedditSGraph.pt"
        },
        "label-smoothing": {
            "values": [0.3]
        },
        "lr": {
            "values": [0.0005, 0.001, 0.002]
        },
        "metric": {
            "values": ["accuracy", "f1"]
        },
        "n-epochs": {
            "value": 1000
        },
        "n-heads": {
            "value": 3
        },
        "n-hidden": {
            "values": [256, 128]
        },
        "n-layers": {
            "values": [2, 3]
        },
        "n-runs": {
            "value": 10
        },
        "selfloop": {
            "value": True
        },
        "undirected": {
            "value": True
        },
        "warmup_epochs": {
            "values": [1, 50]
        }
    },
    "program": "GNN/Library/GAT.py"
}

# 指定项目名称
project_name = "Movies-GNN"

# 创建sweep
sweep_id = wandb.sweep(sweep_config, project=project_name)

# 输出sweep ID
print(f"Sweep ID: {sweep_id}")
