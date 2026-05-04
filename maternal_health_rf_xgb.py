# maternal_health_xai.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# -----------------------
# 1. LOAD DATA
# -----------------------
df = pd.read_csv("Maternal Health Risk Data Set.csv")

print("Initial shape:", df.shape)
print(df.head())

# -----------------------
# 2. DATA CLEANING
# -----------------------
# Remove unrealistic heart rates
df = df[(df['HeartRate'] >= 40) & (df['HeartRate'] <= 200)]

# Encode target labels
le = LabelEncoder()
df['RiskLevelEncoded'] = le.fit_transform(df['RiskLevel'])

# Features and target
X = df.drop(["RiskLevel", "RiskLevelEncoded"], axis=1)
y = df["RiskLevelEncoded"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------
# 3. RANDOM FOREST MODEL
# -----------------------
rf = RandomForestClassifier(n_estimators=200, random_state=0)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Confusion Matrix RF
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Random Forest - Confusion Matrix")
plt.show()

# Feature Importance RF
feat_imp_rf = pd.Series(rf.feature_importances_, index=X.columns)
feat_imp_rf.sort_values().plot(kind="barh", color="teal")
plt.title("Random Forest - Feature Importance")
plt.show()

# -----------------------
# 4. XGBOOST MODEL
# -----------------------
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    objective="multi:softmax",
    num_class=len(np.unique(y)),
    eval_metric="mlogloss",
    random_state=0,
    use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n--- XGBoost Results ---")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Confusion Matrix XGB
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("XGBoost - Confusion Matrix")
plt.show()

# Feature Importance XGB
plot_importance(xgb, max_num_features=10, importance_type="weight")
plt.title("XGBoost - Feature Importance")
plt.show()

# -----------------------
# 5. EXPLAINABLE AI (XAI) with SHAP
# -----------------------
print("\nRunning SHAP explainability for XGBoost...")

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_train)

# Summary Plot (Global feature impact)
shap.summary_plot(shap_values, X_train, feature_names=X.columns)

# Bar Plot (Mean absolute SHAP value)
shap.summary_plot(shap_values, X_train, feature_names=X.columns, plot_type="bar")

# -----------------------
# 6. LOCAL EXPLANATION (Single prediction example)
# -----------------------

sample_index = np.random.randint(0, X_test.shape[0])

sample_index = 0
shap.force_plot(
    explainer.expected_value[0],
    shap_values[0][sample_index, :],
    features=X_test.iloc[sample_index, :],
    feature_names=X_test.columns
)
plt.show()