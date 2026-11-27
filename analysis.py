
# ==========================================
# DATA ANALYSIS AND VISUALIZATION
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR /
    "data" /
    "student-mat.csv"
)

OUTPUT_DIR = (
    BASE_DIR /
    "outputs"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)

# Load dataset
df = pd.read_csv(
    DATA_FILE,
    sep=";"
)

print("Dataset Shape:", df.shape)

print("\nDataset Columns:")

print(df.columns)

# ==========================================
# GRAPH 1: FINAL GRADE DISTRIBUTION
# ==========================================

plt.figure(
    figsize=(8, 5)
)

sns.histplot(

    data=df,

    x="G3",

    bins=20,

    kde=True

)

plt.title(
    "Distribution of Final Grades"
)

plt.xlabel(
    "Final Grade"
)

plt.ylabel(
    "Number of Students"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "grade_distribution.png",
    dpi=300
)

plt.show()


# ==========================================
# GRAPH 2: STUDY TIME VS FINAL GRADE
# ==========================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(

    data=df,

    x="studytime",

    y="G3"

)

plt.title(
    "Study Time vs Final Grade"
)

plt.xlabel(
    "Study Time Category"
)

plt.ylabel(
    "Final Grade"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "studytime_vs_grade.png",
    dpi=300
)

plt.show()


# ==========================================
# GRAPH 3: ABSENCES VS FINAL GRADE
# ==========================================

plt.figure(
    figsize=(8, 5)
)

sns.scatterplot(

    data=df,

    x="absences",

    y="G3"

)

plt.title(
    "Absences vs Final Grade"
)

plt.xlabel(
    "Number of Absences"
)

plt.ylabel(
    "Final Grade"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "absences_vs_grade.png",
    dpi=300
)

plt.show()


# ==========================================
# GRAPH 4: CORRELATION HEATMAP
# ==========================================

numeric_data = (
    df.select_dtypes(
        include="number"
    )
)

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(

    numeric_data.corr(),

    annot=True,

    cmap="coolwarm",

    fmt=".2f"

)

plt.title(
    "Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "correlation_heatmap.png",
    dpi=300
)

plt.show()


# ==========================================
# GRAPH 5: FAILURES VS FINAL GRADE
# ==========================================

plt.figure(
    figsize=(8, 5)
)

sns.boxplot(

    data=df,

    x="failures",

    y="G3"

)

plt.title(
    "Previous Failures vs Final Grade"
)

plt.xlabel(
    "Previous Failures"
)

plt.ylabel(
    "Final Grade"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "failures_vs_grade.png",
    dpi=300
)

plt.show()


# ==========================================
# FINAL
# ==========================================

print("\nAll graphs created!")

print(
    "Check the outputs folder."
)
