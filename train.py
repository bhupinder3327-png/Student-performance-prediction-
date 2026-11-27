
# ==========================================
# STUDENT PERFORMANCE PREDICTION
# MACHINE LEARNING TRAINING FILE
# ==========================================

import pandas as pd
import numpy as np
import joblib
import urllib.request
import zipfile
import io

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "student-mat.csv"


# ==========================================
# 2. DATASET DOWNLOAD
# ==========================================

DATA_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/00320/student.zip"
)


def download_dataset():

    if DATA_FILE.exists():

        print("Dataset already exists!")

        return

    print("Downloading real dataset from UCI...")

    try:

        with urllib.request.urlopen(DATA_URL) as response:

            zip_data = response.read()

        with zipfile.ZipFile(
            io.BytesIO(zip_data)
        ) as zip_file:

            zip_file.extract(
                "student-mat.csv",
                DATA_DIR
            )

        print("Dataset downloaded successfully!")

    except Exception as error:

        print("Dataset download failed!")

        print(error)

        print(
            "Download manually from:"
        )

        print(
            "https://archive.ics.uci.edu/"
            "ml/machine-learning-databases/00320/student.zip"
        )

        raise


# ==========================================
# 3. LOAD DATA
# ==========================================

def load_data():

    download_dataset()

    df = pd.read_csv(
        DATA_FILE,
        sep=";"
    )

    print("\nDATASET LOADED")

    print("Shape:", df.shape)

    print("\nFirst 5 rows:")

    print(df.head())

    return df


# ==========================================
# 4. DATA EXPLORATION
# ==========================================

def explore_data(df):

    print("\n========== DATA INFORMATION ==========")

    print(df.info())

    print("\nMissing Values:")

    print(df.isnull().sum())

    print("\nDuplicate Rows:")

    print(df.duplicated().sum())

    print("\nStatistical Summary:")

    print(df.describe())

    print("\nFinal Grade Distribution:")

    print(df["G3"].value_counts().sort_index())


# ==========================================
# 5. FEATURE AND TARGET SELECTION
# ==========================================

def prepare_data(df):

    # Target variable
    y = df["G3"]

    # Remove target and previous grades
    # G1 and G2 are excluded to avoid
    # direct leakage from earlier grades.
    X = df.drop(
        columns=["G1", "G2", "G3"]
    )

    print("\nFeatures:", X.shape)

    print("Target:", y.shape)

    return X, y


# ==========================================
# 6. PREPROCESSING
# ==========================================

def create_preprocessor(X):

    numerical_features = (
        X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
    )

    categorical_features = (
        X.select_dtypes(
            include=["object"]
        ).columns.tolist()
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )

    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )

    ])

    preprocessor = ColumnTransformer([

        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )

    ])

    return preprocessor


# ==========================================
# 7. CREATE MACHINE LEARNING MODEL
# ==========================================

def create_model(preprocessor):

    random_forest = RandomForestRegressor(

        n_estimators=300,

        random_state=42,

        n_jobs=-1

    )

    model = Pipeline([

        (
            "preprocessor",
            preprocessor
        ),

        (
            "regressor",
            random_forest
        )

    ])

    return model


# ==========================================
# 8. TRAIN AND TEST SPLIT
# ==========================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,

            y,

            test_size=0.20,

            random_state=42

        )
    )

    print("\nTraining Data:", X_train.shape)

    print("Testing Data:", X_test.shape)

    return X_train, X_test, y_train, y_test


# ==========================================
# 9. TRAIN MODEL
# ==========================================

def train_model(
    model,
    X_train,
    y_train
):

    print("\nTraining Random Forest Model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training Completed!")

    return model


# ==========================================
# 10. EVALUATE MODEL
# ==========================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    within_2_marks = (
        np.abs(
            y_test.to_numpy() - predictions
        ) <= 2
    ).mean() * 100

    print("\n================================")

    print("MODEL EVALUATION RESULTS")

    print("================================")

    print(
        "MAE:",
        round(mae, 3)
    )

    print(
        "RMSE:",
        round(rmse, 3)
    )

    print(
        "R2 Score:",
        round(r2, 3)
    )

    print(
        "Predictions within 2 marks:",
        round(within_2_marks, 2),
        "%"
    )

    metrics = {

        "MAE": float(mae),

        "RMSE": float(rmse),

        "R2": float(r2),

        "Within_2_Marks_Percent":
        float(within_2_marks)

    }

    return predictions, metrics


# ==========================================
# 11. SAVE RESULTS
# ==========================================

def save_results(
    y_test,
    predictions,
    metrics
):

    results = pd.DataFrame({

        "Actual_Grade":
        y_test.to_numpy(),

        "Predicted_Grade":
        predictions.round(2),

        "Error":
        np.abs(
            y_test.to_numpy() -
            predictions
        ).round(2)

    })

    results.to_csv(

        OUTPUT_DIR / "predictions.csv",

        index=False

    )

    pd.DataFrame(
        [metrics]
    ).to_csv(

        OUTPUT_DIR / "metrics.csv",

        index=False

    )

    print("\nResults saved!")


# ==========================================
# 12. ACTUAL VS PREDICTED GRAPH
# ==========================================

def create_prediction_graph(
    y_test,
    predictions
):

    plt.figure(
        figsize=(8, 6)
    )

    sns.scatterplot(

        x=y_test,

        y=predictions

    )

    plt.plot(

        [0, 20],

        [0, 20],

        linestyle="--"

    )

    plt.xlabel(
        "Actual Grade"
    )

    plt.ylabel(
        "Predicted Grade"
    )

    plt.title(
        "Actual vs Predicted Student Grades"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /
        "actual_vs_predicted.png",

        dpi=300

    )

    plt.close()

    print(
        "Prediction graph saved!"
    )


# ==========================================
# 13. FEATURE IMPORTANCE GRAPH
# ==========================================

def create_feature_importance_graph(
    model
):

    preprocessor = (
        model.named_steps["preprocessor"]
    )

    random_forest = (
        model.named_steps["regressor"]
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    importance_values = (
        random_forest.feature_importances_
    )

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance_values

    })

    importance_df = (
        importance_df.sort_values(
            by="Importance",
            ascending=False
        )
    )

    top_features = (
        importance_df.head(15)
    )

    top_features.to_csv(

        OUTPUT_DIR /
        "feature_importance.csv",

        index=False

    )

    plt.figure(
        figsize=(10, 7)
    )

    sns.barplot(

        data=top_features,

        x="Importance",

        y="Feature"

    )

    plt.title(
        "Top 15 Important Features"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR /
        "feature_importance.png",

        dpi=300

    )

    plt.close()

    print(
        "Feature importance graph saved!"
    )


# ==========================================
# 14. SAVE MODEL
# ==========================================

def save_model(model):

    model_path = (
        MODEL_DIR /
        "student_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print("\nModel Saved!")

    print(model_path)


# ==========================================
# 15. MAIN FUNCTION
# ==========================================

def main():

    # Load dataset
    df = load_data()

    # Explore data
    explore_data(df)

    # Prepare features and target
    X, y = prepare_data(df)

    # Preprocessing
    preprocessor = create_preprocessor(X)

    # Model
    model = create_model(
        preprocessor
    )

    # Split
    X_train, X_test, y_train, y_test = (
        split_data(X, y)
    )

    # Train
    model = train_model(
        model,
        X_train,
        y_train
    )

    # Evaluate
    predictions, metrics = (
        evaluate_model(
            model,
            X_test,
            y_test
        )
    )

    # Save evaluation
    save_results(
        y_test,
        predictions,
        metrics
    )

    # Graphs
    create_prediction_graph(
        y_test,
        predictions
    )

    create_feature_importance_graph(
        model
    )

    # Save model
    save_model(model)

    print("\n================================")

    print("PROJECT COMPLETED SUCCESSFULLY")

    print("================================")


if __name__ == "__main__":

    main()
