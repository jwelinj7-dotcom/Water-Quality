
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


# Load dataset
df = pd.read_excel("Water quality dataset.xlsx")

np.random.seed(42)

# Add small noise to parameters
df["pH"] = df["pH"] + np.random.normal(0, 0.25, len(df))
df["Turbidity"] = df["Turbidity"] + np.random.normal(0, 0.7, len(df))
df["Total Hardness"] = df["Total Hardness"] + np.random.normal(0, 40, len(df))

# Keep realistic limits
df["pH"] = df["pH"].clip(6, 9)
df["Turbidity"] = df["Turbidity"].clip(0.3, 10)
df["Total Hardness"] = df["Total Hardness"].clip(50, 600)

# Assign quality with overlapping ranges


def assign_quality(row):

    score = (
        abs(row["pH"] - 7.2) * 1.2 +
        row["Turbidity"] * 0.6 +
        row["Total Hardness"] / 250
    )

    r = np.random.rand()

    if score < 2.5:
        return "Safe" if r > 0.15 else "Moderate"
    elif score < 4:
        return "Moderate" if r > 0.2 else np.random.choice(["Safe", "Unsafe"])
    else:
        return "Unsafe" if r > 0.15 else "Moderate"


df["Quality"] = df.apply(assign_quality, axis=1)

# Save updated dataset
df.to_excel("water_quality_updated.xlsx", index=False)


# Clean dataset


df = df.drop_duplicates()
df = df.dropna()


# Define features and label


X = df.drop(columns=['Sample_ID', 'Quality'])
y = df['Quality']


# Encode the target values(safe -> 0,Moderate -> 1,Unsafe -> 2)


le = LabelEncoder()
y = le.fit_transform(y)


# Split dataset


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# Create the model


rf_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)


# Train the model


rf_model.fit(X_train, y_train)


# Make Predictions


y_pred = rf_model.predict(X_test)
y_pred


# Check Model Accuracy


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
plt.bar(["Accuracy"], [accuracy])

plt.title("Model Accuracy")
plt.ylim(0, 1)

plt.show()

# Confusion Matrix(correct VS incorrect predictions)


cm = confusion_matrix(y_test, y_pred)
print(cm)


# Predicting the state of new sample


new_sample = pd.DataFrame([[6.5, 0.5, 25]], columns=X.columns)
prediction = rf_model.predict(new_sample)
print(prediction)
prediction = rf_model.predict(new_sample)
result = le.inverse_transform(prediction)
print("Water Quality:", result[0])


print(classification_report(y_test, y_pred))


# Feature Distribution


sns.histplot(df['pH'], kde=True)
plt.title("pH Distribution")
plt.show()

sns.histplot(df['Turbidity'], kde=True)
plt.title("Turbidity Distribution")
plt.show()

sns.histplot(df['Total Hardness'], kde=True)
plt.title("Total Hardness Distribution")
plt.show()


# Feature Importance


importances = rf_model.feature_importances_

feature_names = X.columns

plt.bar(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Parameters")
plt.ylabel("Importance")
plt.show()


df['WQI'] = (
    (df['pH']/8.5)*0.3 +
    (df['Turbidity']/5)*0.4 +
    (df['Total Hardness']/600)*0.3
)


# Water Quality Index Calculation


sns.histplot(df['WQI'])
plt.title("Water Quality Index Distribution")
plt.show()


# Save the trained Model



joblib.dump(rf_model, "water_quality_model.pkl")
joblib.dump(le, "label_encoder.pkl")
