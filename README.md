Detailed Code Explanation
1. Main Execution (run.py & pipeline/training_pipeline.py)
run.py : This is the main entry point to start the entire training pipeline.

Purpose: It orchestrates all the steps from configuration generation to model training.

Key Logic:

It initializes a custom logger.

It calls generate_config() to automatically create the config.yaml file based on the columns of the raw dataset. This is a powerful feature that makes the project adaptable to new datasets.

It then sequentially runs the Data Ingestion, Data Preprocessing, and Model Training steps by creating instances of their respective classes and calling their run() or process() methods.

Includes robust try...except blocks using the CustomException class for better error handling.

pipeline/training_pipeline.py : This script provides a slightly simpler way to run the pipeline, often used for testing or specific runs without the config generation step.

2. Configuration (config/)
This directory centralizes all project configurations.

paths_config.py : Defines all important file and directory paths used throughout the project, such as paths to raw data, processed data, and the final model. This avoids hardcoding paths in multiple files.

buildconfig.py : Contains the generate_config function. It reads the raw dataset, automatically identifies categorical and numerical columns, and writes them into a config.yaml file.

config.yaml : The central configuration file. It stores parameters for data ingestion (e.g., train/test split ratio), data processing (column types, skewness threshold, number of features to select), and other settings.

model_params.py : Defines the hyperparameter search space for model tuning. It specifies the parameters for the LightGBM model and the settings for RandomizedSearchCV.

3. Data Ingestion (src/data_ingestion.py)
Purpose: To load the initial dataset and split it for training and testing.

DataIngestion Class :

__init__: Initializes with the configuration from config.yaml. Creates the artifacts/raw directory if it doesn't exist.

load_input_data(): Reads the raw CSV file from its path into a pandas DataFrame.

split_data(): Uses train_test_split from scikit-learn to divide the data into training and testing sets based on the ratio defined in config.yaml. It then saves these two new datasets (train.csv and test.csv) into the artifacts/raw directory.

run(): The main method that executes the loading and splitting steps in sequence.

4. Data Preprocessing (src/data_preprocessing.py)
Purpose: This is the most critical data transformation step to prepare the data for modeling.

DataProcessor Class :

preprocess_data():

Cleaning: Drops unnecessary columns and duplicate rows.

Label Encoding: Converts categorical string columns (like type_of_meal_plan) into numerical representations using LabelEncoder. This is necessary for most machine learning models.

Skewness Handling: Calculates the skewness of numerical columns and applies a log transformation (np.log1p) to features with high skewness. This helps models that assume a normal distribution of data.

balance_data():

Addresses class imbalance using SMOTE (Synthetic Minority Over-sampling Technique). In cancellation prediction, it's common for one class (e.g., "Not Canceled") to be much larger than the other ("Canceled"). SMOTE creates synthetic data points for the minority class to balance the dataset, which helps prevent the model from being biased towards the majority class.

select_features():

Uses a RandomForestClassifier to determine feature importance.

It then selects the top 10 features (as defined in config.yaml) and discards the rest. This simplifies the model, reduces training time, and can improve performance by removing noise.

process(): Orchestrates all the above steps for both the training and testing datasets and saves the final, clean dataframes to the artifacts/processed directory.

5. Model Training and Tuning (src/ModelTuner_Classification.py & src/model_training.py)
This is where the machine learning happens.

src/ModelTuner_Classification.py : A powerful, reusable class for evaluating and tuning a wide range of classifiers.

Purpose: To systematically find the best algorithm and hyperparameters for the given dataset.

Key Methods:

_get_models(): Contains a dictionary of different scikit-learn models to be evaluated (e.g., RandomForest, XGBoost, LightGBM).

evaluate_models(): Iterates through all models, trains them on default parameters, and generates a performance report (Accuracy, Precision, Recall, F1 Score). It returns a sorted dataframe, making it easy to see which model performed best.

tune_model(): Takes the name of a model and performs hyperparameter tuning using RandomizedSearchCV. This method intelligently searches for the best combination of parameters from the search space defined in config/model_params.py.

get_best_model(): A wrapper method that first runs evaluate_models() to find the best baseline model and then automatically runs tune_model() on that best model.

src/model_training.py : This script uses the ModelTuner to train the final model.

ModelTraining Class:

load_and_split_data(): Loads the processed data and separates it into features (X) and target (y).

save_model(): Saves the final trained model object to a file (.pkl) using joblib.

run():

Initializes an MLflow run to track the experiment.

Logs the dataset artifacts to MLflow.

Calls the get_best_model() method from the ModelTuner class to find, train, and tune the best model.

Saves the final tuned model to the artifacts/models directory.

Logs the best parameters and performance metrics to MLflow, providing a complete record of the training run.

6. Web Application (application.py)
Purpose: To provide a user-friendly interface for interacting with the trained model.

Framework: Uses Flask, a lightweight Python web framework.

Key Logic:

It loads the saved joblib model file from the artifacts/models directory when the application starts.

It defines a single route (/) that handles both GET (displaying the page) and POST (handling form submission) requests.

It renders the templates/index.html file, which contains a form for the user to input reservation details.

When the user submits the form, the app collects the data, converts it into a NumPy array with the correct format for the model, and calls loaded_model.predict().

The prediction result (0 for "Not Canceled", 1 for "Canceled") is sent back to the index.html template and displayed to the user.

7. DevOps and Utilities
utils/common_functions.py : Contains helper functions like read_yaml and load_data to avoid code duplication.

src/logger.py & src/custom_exception.py: Provide a robust logging and exception handling framework. All important events and errors are logged to a file in the logs/ directory, making debugging much easier.

Dockerfile : Contains instructions to package the entire application (including the Flask app and the trained model) into a Docker container. This makes the application portable and easy to deploy on any server.

Jenkinsfile : Defines the CI/CD pipeline. It likely contains stages to automatically build the Docker image and deploy the container whenever changes are pushed to the Git repository.

requirements.txt : Lists all the Python libraries required to run the project. This ensures a consistent environment.
