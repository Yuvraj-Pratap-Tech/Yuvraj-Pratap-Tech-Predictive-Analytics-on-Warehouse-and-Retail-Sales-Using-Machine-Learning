# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:28:09 2026

@author: LENOVO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

df = pd.read_csv("D:/3rd YEAR A.P.S/A.P. SEM1/INT234 PREDICTIVE ANALYTICS/Warehouse and Retail Sales.csv")
df.columns = df.columns.str.strip().str.upper()

print("First Five Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows")
print(df.duplicated().sum())

numeric_cols = [
    "YEAR",
    "MONTH",
    "RETAIL SALES",
    "RETAIL TRANSFERS",
    "WAREHOUSE SALES"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\nNumerical Summary")
print(df[numeric_cols].describe())

df[numeric_cols].hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df["RETAIL SALES"].dropna(), bins=40, kde=True)
plt.title("Distribution of Retail Sales")
plt.show()

plot_df = df.dropna(subset=["WAREHOUSE SALES", "RETAIL SALES"])
plot_df = plot_df.sample(min(5000, len(plot_df)), random_state=42)

plt.figure(figsize=(8, 5))
sns.scatterplot(data=plot_df, x="WAREHOUSE SALES", y="RETAIL SALES")
plt.title("Warehouse Sales vs Retail Sales")
plt.show()

print("\nItem Type Distribution")
print(df["ITEM TYPE"].value_counts().head(10))

print("\nSupplier Distribution")
print(df["SUPPLIER"].value_counts().head(10))

df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()
df = df.dropna(subset=["RETAIL SALES"])

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    mode_values = df[col].mode()
    df[col] = df[col].fillna(
        mode_values.iloc[0] if not mode_values.empty else "Unknown"
    )

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df = df.dropna(subset=numeric_cols)

print("\nMissing Values After Preprocessing")
print(df.isnull().sum())

print("\nFinal Dataset Shape")
print(df.shape)

X = df[["YEAR", "MONTH", "WAREHOUSE SALES", "RETAIL TRANSFERS"]]
y = df["RETAIL SALES"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n========== POLYNOMIAL REGRESSION ==========")

model_poly = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)

model_poly.fit(X_train, y_train)
y_pred_poly = model_poly.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred_poly))
print("MSE:", mean_squared_error(y_test, y_pred_poly))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_poly)))
print("R2 Score:", r2_score(y_test, y_pred_poly))


# Graph: Actual vs Predicted Values
plt.figure(figsize=(8, 5))

plt.scatter(y_test, y_pred_poly, alpha=0.7, label="Predicted Values")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red',
    linestyle='--',
    label="Perfect Prediction"
)

plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Polynomial Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

print("\n========== NAIVE BAYES ==========")

y_class = (df["RETAIL SALES"] >= df["RETAIL SALES"].median()).astype(int)

X_nb = df[["YEAR", "MONTH", "RETAIL TRANSFERS", "WAREHOUSE SALES"]]

X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
    X_nb, y_class, test_size=0.2, random_state=42, stratify=y_class
)

scaler_nb = StandardScaler()
X_train_nb_scaled = scaler_nb.fit_transform(X_train_nb)
X_test_nb_scaled = scaler_nb.transform(X_test_nb)

model_nb = GaussianNB()
model_nb.fit(X_train_nb_scaled, y_train_nb)
y_pred_nb = model_nb.predict(X_test_nb_scaled)

print("Accuracy:", accuracy_score(y_test_nb, y_pred_nb))
print("Precision:", precision_score(y_test_nb, y_pred_nb, zero_division=0))
print("Recall:", recall_score(y_test_nb, y_pred_nb, zero_division=0))
print("F1 Score:", f1_score(y_test_nb, y_pred_nb, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test_nb, y_pred_nb))
print(classification_report(y_test_nb, y_pred_nb, zero_division=0))

cm = confusion_matrix(y_test_nb, y_pred_nb)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Low Retail Sales", "High Retail Sales"],
    yticklabels=["Low Retail Sales", "High Retail Sales"]
)

plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

print("\n========== SIMPLE LINEAR REGRESSION ==========")

X_slr = df[["WAREHOUSE SALES"]]
y_slr = df["RETAIL SALES"]

X_train_slr, X_test_slr, y_train_slr, y_test_slr = train_test_split(
    X_slr, y_slr, test_size=0.2, random_state=42
)

model_slr = LinearRegression()
model_slr.fit(X_train_slr, y_train_slr)
y_pred_slr = model_slr.predict(X_test_slr)

print("Coefficient:", model_slr.coef_[0])
print("Intercept:", model_slr.intercept_)
print("MAE:", mean_absolute_error(y_test_slr, y_pred_slr))
print("MSE:", mean_squared_error(y_test_slr, y_pred_slr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_slr, y_pred_slr)))
print("R2 Score:", r2_score(y_test_slr, y_pred_slr))

plt.figure(figsize=(8, 5))
plt.scatter(X_test_slr["WAREHOUSE SALES"], y_test_slr, alpha=0.3)
plt.plot(
    X_test_slr["WAREHOUSE SALES"],
    y_pred_slr,
    color="red"
)
plt.xlabel("Warehouse Sales")
plt.ylabel("Retail Sales")
plt.title("Simple Linear Regression")
plt.show()

print("\n========== SUPPORT VECTOR REGRESSION ==========")

svr_df = df.sample(min(10000, len(df)), random_state=42)

X_svr = svr_df[["YEAR", "MONTH", "WAREHOUSE SALES", "RETAIL TRANSFERS"]]
y_svr = svr_df["RETAIL SALES"]

X_train_svr, X_test_svr, y_train_svr, y_test_svr = train_test_split(
    X_svr, y_svr, test_size=0.2, random_state=42
)

model_svr = make_pipeline(
    StandardScaler(),
    SVR(kernel="rbf", C=1.0, epsilon=0.1)
)

model_svr.fit(X_train_svr, y_train_svr)
y_pred_svr = model_svr.predict(X_test_svr)

print("MAE:", mean_absolute_error(y_test_svr, y_pred_svr))
print("MSE:", mean_squared_error(y_test_svr, y_pred_svr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_svr, y_pred_svr)))
print("R2 Score:", r2_score(y_test_svr, y_pred_svr))
plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_svr,
    y_pred_svr,
    alpha=0.7,
    label="Predicted Values"
)

plt.plot(
    [y_test_svr.min(), y_test_svr.max()],
    [y_test_svr.min(), y_test_svr.max()],
    color='red',
    linestyle='--',
    label="Perfect Prediction"
)

plt.xlabel("Actual Retail Sales")
plt.ylabel("Predicted Retail Sales")
plt.title("Support Vector Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()

print("\n========== MULTIPLE LINEAR REGRESSION ==========")

X_mlr = df[["YEAR", "MONTH", "WAREHOUSE SALES", "RETAIL TRANSFERS"]]
y_mlr = df["RETAIL SALES"]

X_train_mlr, X_test_mlr, y_train_mlr, y_test_mlr = train_test_split(
    X_mlr, y_mlr, test_size=0.2, random_state=42
)

model_mlr = LinearRegression()
model_mlr.fit(X_train_mlr, y_train_mlr)
y_pred_mlr = model_mlr.predict(X_test_mlr)

print("Coefficients:", model_mlr.coef_)
print("Intercept:", model_mlr.intercept_)
print("MAE:", mean_absolute_error(y_test_mlr, y_pred_mlr))
print("MSE:", mean_squared_error(y_test_mlr, y_pred_mlr))
print("RMSE:", np.sqrt(mean_squared_error(y_test_mlr, y_pred_mlr)))
print("R2 Score:", r2_score(y_test_mlr, y_pred_mlr))

plt.figure(figsize=(8, 5))

plt.scatter(
    y_test_mlr,
    y_pred_mlr,
    alpha=0.7,
    label="Predicted Values"
)

plt.plot(
    [y_test_mlr.min(), y_test_mlr.max()],
    [y_test_mlr.min(), y_test_mlr.max()],
    color='red',
    linestyle='--',
    label="Perfect Prediction"
)

plt.xlabel("Actual Retail Sales")
plt.ylabel("Predicted Retail Sales")
plt.title("Multiple Linear Regression: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.show()
