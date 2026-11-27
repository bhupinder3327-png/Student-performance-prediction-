
# ==========================================
# STUDENT PERFORMANCE PREDICTION APP
# STREAMLIT APPLICATION
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

from pathlib import Path


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "student-mat.csv"

MODEL_FILE = BASE_DIR / "models" / "student_model.pkl"

OUTPUT_DIR = BASE_DIR / "outputs"


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🎓 Student Performance Prediction")

st.subheader("Machine Learning Regression Application")

st.write(
    "Predict a student's final Mathematics grade "
    "using student-related information."
)

st.divider()


# ==========================================
# CHECK FILES
# ==========================================

if not DATA_FILE.exists():

    st.error(
        "Dataset not found! Please put "
        "student-mat.csv inside the data folder."
    )

    st.stop()


if not MODEL_FILE.exists():

    st.error(
        "Trained model not found! "
        "First run python train.py"
    )

    st.stop()


# ==========================================
# LOAD DATA AND MODEL
# ==========================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        DATA_FILE,
        sep=";"
    )


@st.cache_resource
def load_model():

    return joblib.load(MODEL_FILE)


df = load_dataset()

model = load_model()


# ==========================================
# INPUT FEATURES
# ==========================================

X = df.drop(
    columns=["G1", "G2", "G3"]
)

st.info(
    "The model does not use G1 or G2. "
    "It predicts performance using other "
    "student-related features."
)


# ==========================================
# STUDENT INFORMATION FORM
# ==========================================

st.header("📝 Student Information")

st.write("Enter the student's details below.")

input_values = {}

with st.form("student_prediction_form"):

    columns = st.columns(2)

    for index, feature in enumerate(X.columns):

        current_column = columns[index % 2]

        with current_column:

            # ==================================
            # CATEGORICAL FEATURES
            # ==================================

            if (
                pd.api.types.is_object_dtype(X[feature])
                or pd.api.types.is_string_dtype(X[feature])
                or isinstance(X[feature].dtype, pd.CategoricalDtype)
            ):

                choices = sorted(
                    X[feature]
                    .dropna()
                    .unique()
                    .tolist()
                )

                input_values[feature] = st.selectbox(
                    label=feature.replace("_", " ").title(),
                    options=choices,
                    key=f"input_{feature}"
                )

            # ==================================
            # NUMERICAL FEATURES
            # ==================================

            elif pd.api.types.is_numeric_dtype(X[feature]):

                minimum = float(X[feature].min())

                maximum = float(X[feature].max())

                default_value = float(X[feature].median())

                # Integer numerical features
                if pd.api.types.is_integer_dtype(X[feature]):

                    input_values[feature] = st.slider(
                        label=feature.replace("_", " ").title(),
                        min_value=int(minimum),
                        max_value=int(maximum),
                        value=int(default_value),
                        step=1,
                        key=f"input_{feature}"
                    )

                # Decimal numerical features
                else:

                    input_values[feature] = st.slider(
                        label=feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default_value,
                        step=0.1,
                        key=f"input_{feature}"
                    )

            else:

                st.warning(
                    f"Unsupported data type: {feature}"
                )


    # ==================================
    # SUBMIT BUTTON
    # ==================================

    predict_button = st.form_submit_button(
        "🚀 Predict Final Grade",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if predict_button:

    try:

        # Create input dataframe
        input_df = pd.DataFrame(
            [input_values]
        )

        # Keep correct feature order
        input_df = input_df[X.columns]

        # Make prediction
        prediction = model.predict(
            input_df
        )[0]

        # Keep grade within 0 to 20
        prediction = np.clip(
            prediction,
            0,
            20
        )

        # ==================================
        # DISPLAY RESULTS
        # ==================================

        st.divider()

        st.header("🎯 Prediction Result")

        st.success(
            "Prediction completed successfully!"
        )

        st.metric(
            label="Predicted Final Grade",
            value=f"{prediction:.2f} / 20"
        )

        # ==================================
        # INTERPRETATION
        # ==================================

        if prediction >= 16:

            st.success(
                "Higher predicted grade range."
            )

        elif prediction >= 10:

            st.info(
                "Passing-range estimate."
            )

        else:

            st.warning(
                "Lower predicted grade range. "
                "Additional academic support may be useful."
            )

        # ==================================
        # PROGRESS BAR
        # ==================================

        st.write("Estimated Grade Visualization")

        st.progress(
            int(prediction / 20 * 100)
        )

        # ==================================
        # DISCLAIMER
        # ==================================

        st.caption(
            "This prediction is based on historical data. "
            "It is an estimate and not a guarantee "
            "of actual exam performance."
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )


# ==========================================
# MODEL INFORMATION
# ==========================================

st.divider()

st.header("🤖 About the Machine Learning Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Algorithm",
    "Random Forest"
)

col2.metric(
    "Problem Type",
    "Regression"
)

col3.metric(
    "Target Variable",
    "G3"
)


st.markdown(
    """
    ### Project Workflow

    1. Student information is entered.
    2. Input data is converted into a DataFrame.
    3. Data is passed to the trained ML pipeline.
    4. Random Forest predicts the final grade.
    5. The predicted grade is displayed.
    """
)


# ==========================================
# MODEL PERFORMANCE
# ==========================================

metrics_file = OUTPUT_DIR / "metrics.csv"

if metrics_file.exists():

    metrics = pd.read_csv(metrics_file)

    st.divider()

    st.header("📊 Model Performance")

    metric_columns = st.columns(3)

    if "MAE" in metrics.columns:

        metric_columns[0].metric(
            "MAE",
            f"{metrics['MAE'].iloc[0]:.2f}"
        )

    if "RMSE" in metrics.columns:

        metric_columns[1].metric(
            "RMSE",
            f"{metrics['RMSE'].iloc[0]:.2f}"
        )

    if "R2" in metrics.columns:

        metric_columns[2].metric(
            "R² Score",
            f"{metrics['R2'].iloc[0]:.2f}"
        )


# ==========================================
# PROJECT VISUALIZATIONS
# ==========================================

st.divider()

st.header("📈 Project Visualizations")

graph1 = OUTPUT_DIR / "actual_vs_predicted.png"

graph2 = OUTPUT_DIR / "feature_importance.png"


if graph1.exists():

    st.subheader("Actual vs Predicted Grades")

    st.image(
        str(graph1),
        use_container_width=True
    )


if graph2.exists():

    st.subheader("Feature Importance")

    st.image(
        str(graph2),
        use_container_width=True
    )


if not graph1.exists() and not graph2.exists():

    st.info(
        "Graphs not found. Run train.py first "
        "to generate project visualizations."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Student Performance Prediction | "
    "Educational Machine Learning Project"
)

st.caption(
    "Dataset: UCI Machine Learning Repository"
)
