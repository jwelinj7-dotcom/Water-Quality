import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


df = pd.read_csv("Water quality dataset.csv")

np.random.seed(42)


df["pH"] += np.random.normal(0, 0.3, len(df))
df["Turbidity"] += np.random.normal(0, 0.8, len(df))
df["Total_Hardness"] += np.random.normal(0, 30, len(df))


df["pH"] = df["pH"].clip(5.5, 9.5)
df["Turbidity"] = df["Turbidity"].clip(0, 10)
df["Total_Hardness"] = df["Total_Hardness"].clip(50, 600)


df = df.drop_duplicates()
df = df.dropna()


df = df.sample(frac=1, random_state=42).reset_index(drop=True)


X = df.drop(columns=['Sample_ID', 'Quality'])
y = df['Quality']


le = LabelEncoder()
y = le.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight={0:1, 1:1, 2:2},
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nRandom Forest Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


importances = rf_model.feature_importances_

plt.figure()
plt.bar(X.columns, importances)
plt.title("Feature Importance")
plt.xlabel("Parameters")
plt.ylabel("Importance")
plt.show()


plt.figure()
plt.bar(["Random Forest"], [accuracy])
plt.ylim(0, 1)
plt.title("Random Forest Accuracy")
plt.show()


models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB()
}

results = {}


results["Random Forest"] = accuracy

print("\n--- Model Comparison ---\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"{name}: {acc:.3f}")


plt.figure()

names = list(results.keys())
values = list(results.values())

plt.bar(names, values)
plt.title("Algorithm Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=30)

plt.show()


new_sample = pd.DataFrame([[6.6, 2.0, 180]], columns=X.columns)

prediction = rf_model.predict(new_sample)
result = le.inverse_transform(prediction)

print("\nNew Sample Prediction:", result[0])


joblib.dump(rf_model, "water_quality_model.pkl")
joblib.dump(le, "label_encoder.pkl")
