import pandas as pd
import yaml
import glob
import os




def generate_config(raw_df):
    
    # Step 1: Load dataset
    df= pd.DataFrame(raw_df)

    # Step 2: Separate categorical and numerical columns
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Step 3: Build config dictionary
    config = {
        "data_ingestion": {
            "bucket_name": "AKS_bucket9789",
            "bucket_file_name": "Ashwani shukla",
            "train_ratio": 0.8,
            "test_ratio": 0.2
        },
        "data_processing": {
            "categorical_columns": categorical_columns,
            "numerical_columns": numerical_columns,
            "skewness_threshold": 5,
            "no_of_features": 10
        }
    }

    # Step 4: Save to YAML file
    with open("config/config.yaml", "w") as file:
        yaml.dump(config, file, sort_keys=False, default_flow_style=False)

    print("✅ config.yml generated successfully!")


