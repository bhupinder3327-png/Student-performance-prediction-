# 🎓 Student Performance Prediction Using Machine Learning

### AI-Based Academic Performance Prediction System

**Developed by:** Bhupinder Singh & Madhav  
**University:** Shoolini University  
**Department:** [Enter Department Name]  
**Project Type:** Machine Learning Regression  
**Algorithm:** Random Forest Regressor  
**Programming Language:** Python  
**Web Framework:** Streamlit  

---

## 📌 Project Overview

Student Performance Prediction is a Machine Learning project designed to predict a student's expected academic score based on important academic and lifestyle factors.

The system uses a Random Forest Regression model to analyze student information and predict an expected final score out of 100.

The project provides an interactive web application developed using Streamlit, where users can enter student details and receive an AI-generated prediction along with personalized suggestions for academic improvement.

This project demonstrates the complete Machine Learning workflow, including data generation, model training, testing, evaluation, model saving, and deployment through a web application.

---

## 🎯 Project Objectives

- Predict students' expected academic performance.
- Understand the relationship between academic habits and performance.
- Apply Machine Learning techniques to a real-world problem.
- Implement a supervised learning regression model.
- Build an interactive user-friendly application.
- Display model evaluation metrics.
- Provide suggestions for improving academic habits.

---

## 🚀 Features

- Simple and interactive web interface.
- Student performance prediction.
- Six student input parameters.
- Random Forest Regression model.
- Prediction score between 0 and 100.
- Performance category classification.
- Personalized academic improvement suggestions.
- Model evaluation metrics.
- Feature importance visualization.
- Synthetic dataset generation.
- Model saving and loading using Joblib.

---

## 🧑‍🎓 Student Information Used for Prediction

The model uses six input features.

| Feature | Description | Range |
|---|---|---|
| Attendance Rate | Percentage of classes attended | 0–100% |
| Weekly Study Hours | Total study hours per week | 0–40 hours |
| Assignment Completion | Percentage of assignments completed | 0–100% |
| Previous Exam Score | Previous academic exam score | 0–100 |
| Sleep Hours | Average sleep hours per day | 0–12 hours |
| Class Participation | Participation in class activities | 0–10 |

### Target Variable

**Final Score**

The model predicts the student's expected academic score between 0 and 100.

---

## 🧠 Machine Learning Concepts Used

### Artificial Intelligence

Artificial Intelligence is the technology that enables machines to perform tasks that typically require human intelligence.

### Machine Learning

Machine Learning is a subset of Artificial Intelligence that allows computers to learn patterns from data and make predictions.

### Supervised Learning

Supervised Learning trains a model using examples containing inputs and known outputs.

Our model learns from student-related features and their corresponding final scores.

### Regression

Regression is used when the target variable is numerical.

Our project uses Regression because it predicts a continuous numerical score.

---

## 🌳 Algorithm: Random Forest Regressor

Random Forest is an ensemble Machine Learning algorithm that combines multiple Decision Trees.

For regression, the algorithm combines predictions from different trees and calculates an average prediction.

### Why Random Forest?

- Can learn nonlinear patterns.
- Combines multiple decision trees.
- Reduces the limitations of individual decision trees.
- Works well for many structured datasets.
- Provides feature importance information.

### Model Configuration

```python
RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
```

| Parameter | Description |
|---|---|
| `n_estimators=200` | Number of decision trees |
| `max_depth=12` | Maximum depth of each tree |
| `random_state=42` | Reproducibility |
| `n_jobs=-1` | Uses available CPU cores |

---

## 📊 Dataset

This project uses a **synthetically generated dataset** created using Python and NumPy.

The dataset contains 1,500 sample student records.

Synthetic data is generated to demonstrate the Machine Learning pipeline without requiring real student records or exposing personal information.

### Dataset Features

- Attendance Rate
- Weekly Study Hours
- Assignment Completion
- Previous Score
- Sleep Hours
- Class Participation
- Final Score

### Important Disclaimer

The synthetic dataset is created using predefined mathematical relationships and random variation.

Therefore, model evaluation scores demonstrate the functionality of the prototype and should not be interpreted as proof of real-world predictive accuracy.

Future versions should use real, anonymized student data collected with appropriate permission.

---

## ⚙️ Project Workflow

1. Generate synthetic student dataset.
2. Separate input features and target variable.
3. Split data into training and testing datasets.
4. Create Random Forest Regressor.
5. Train the model using training data.
6. Predict scores on test data.
7. Evaluate model performance.
8. Save trained model.
9. Create feature importance visualization.
10. Load trained model into Streamlit.
11. Accept user input.
12. Predict expected student score.
13. Display results and improvement suggestions.

---

## 📂 Project Structure

```text
StudentPerformancePrediction/
│
├── app.py
│   └── Streamlit web application
│
├── train.py
│   └── Dataset generation and model training
│
├── requirements.txt
│   └── Required Python libraries
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Files excluded from GitHub
│
├── data/
│   └── student_data.csv
│       └── Generated synthetic dataset
│
├── models/
│   └── student_model.pkl
│       └── Saved trained model
│
└── outputs/
    ├── metrics.csv
    ├── feature_importance.csv
    └── feature_importance.png
```

---

## 💻 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Dataset handling |
| NumPy | Numerical calculations |
| Scikit-learn | Machine Learning |
| Random Forest | Prediction algorithm |
| Streamlit | Web application |
| Joblib | Saving and loading models |
| Matplotlib | Data visualization |
| Git | Version control |
| GitHub | Project repository |

---

## 📦 Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Student-Performance-Prediction.git
```

### Step 2: Navigate to the project folder

```bash
cd Student-Performance-Prediction
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Machine Learning Model

Run:

```bash
python train.py
```

The training program will:

- Generate a synthetic dataset.
- Split the dataset into training and testing data.
- Train the Random Forest model.
- Evaluate model performance.
- Save the trained model.
- Generate feature importance information.

### Generated Files

```text
data/student_data.csv
models/student_model.pkl
outputs/metrics.csv
outputs/feature_importance.csv
outputs/feature_importance.png
```

---

## ▶️ Running the Application

After training the model, run:

```bash
python -m streamlit run app.py
```

The application will open in your web browser.

Local URL:

```text
http://localhost:8501
```

---

## 📈 Model Evaluation

The model is evaluated using the following metrics:

### 1. Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual values and predicted values.

Lower MAE indicates better predictions.

### 2. Root Mean Squared Error (RMSE)

RMSE measures prediction errors and penalizes larger errors more heavily.

Lower RMSE indicates better performance.

### 3. R² Score

R² measures how well the model explains variations in the target variable relative to a baseline.

A score closer to 1 generally indicates better fit on the evaluated dataset.

### Evaluation Results

Evaluation values are generated when `train.py` runs.

The results are saved in:

```text
outputs/metrics.csv
```

---

## 📊 Feature Importance

Feature importance identifies which features the trained Random Forest model relied on most when making predictions.

The project generates:

```text
outputs/feature_importance.csv
outputs/feature_importance.png
```

The feature importance graph provides a visual comparison of the relative importance of different inputs.

**Note:** Feature importance does not prove causation. It shows how the trained model uses features within this dataset.

---

## 🖥️ Application Interface

The Streamlit application allows users to enter:

- Attendance rate
- Weekly study hours
- Assignment completion
- Previous exam score
- Sleep hours
- Class participation

After clicking the prediction button, the application displays:

- Expected final score
- Performance category
- Score visualization
- Personalized improvement suggestions
- Model evaluation metrics
- Feature importance visualization

### Performance Categories

| Predicted Score | Category |
|---|---|
| 85–100 | Excellent |
| 70–84.99 | Good |
| 50–69.99 | Average |
| Below 50 | Needs Improvement |

These categories are illustrative thresholds used by the application, not official academic grading standards.

---

## 🔮 Future Scope

The project can be improved in the future by:

- Using real anonymized student datasets.
- Comparing Random Forest with Linear Regression.
- Comparing Random Forest with Gradient Boosting.
- Adding subject-wise prediction.
- Adding attendance trend analysis.
- Improving data collection.
- Adding user authentication.
- Storing student prediction history.
- Deploying the application online.
- Integrating a database.
- Improving model performance using hyperparameter tuning.

---

## ⚠️ Limitations

- The dataset is synthetically generated.
- Predictions are not validated using real student data.
- Model predictions may not represent actual academic outcomes.
- Student performance depends on many factors not included in this project.
- Feature importance does not establish causal relationships.
- The prediction should not be used for high-stakes academic decisions.

---

## 👨‍💻 Developed By

**Bhupinder Singh**  
**Madhav**

Shoolini University

Machine Learning Academic Project

---

## 📚 Learning Outcomes

Through this project, we learned:

- Fundamentals of Artificial Intelligence.
- Machine Learning and Supervised Learning.
- Regression problems.
- Feature selection and target variables.
- Dataset generation.
- Training and testing.
- Random Forest algorithm.
- Model evaluation metrics.
- Python libraries.
- Streamlit web application development.
- Model saving and loading.
- Git and GitHub version control.

---

## 🏁 Conclusion

Student Performance Prediction demonstrates the practical implementation of Machine Learning for academic performance estimation.

The project combines Python, Scikit-learn, Random Forest Regression, and Streamlit to build a working prediction prototype.

It provides an educational demonstration of how student-related inputs can be processed by a Machine Learning model to generate predictions.

The project also highlights the importance of data quality, model evaluation, privacy, and real-world validation in Machine Learning applications.

---

## 📄 License

This project is created for educational and academic purposes.

---

⭐ If you find this project useful, feel free to explore and improve it.
