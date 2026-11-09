
import pandas as pd
#DATA UPLOADING AND CLEANING
# Load the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

# Display missing values before cleaning
print("Missing values before cleaning:")
print(df.isnull().sum())

# Remove columns not used in our first model
df = df.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"]
)

# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the most common port
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Check missing values after cleaning
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Display the cleaned dataset
print("\nCleaned dataset:")
print(df.head())

print("\nDataset shape after cleaning:")
print(df.shape)


#VISUALIZATIONS
import matplotlib.pyplot as plt
import seaborn as sns
# Graph 1: Number of passengers who survived
sns.countplot(data=df, x="Survived")
plt.title("Titanic Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.show()

# Graph 2: Survival based on gender
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# Graph 3: Survival based on passenger class
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

#MACHINE LEARNING

# Step 5: Prepare data for Machine Learning

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True,
    dtype=int
)

print("\nDataset after encoding:")
print(df.head())

print("\nUpdated column names:")
print(df.columns.tolist())

#Step 4 : Separate input features and target
#X contains passengers information
X = df.drop("Survived", axis=1)

#y contains   the actual survival outcome
y = df["Survived"]

print("\n Input features shape:", X.shape)
print("Target shape:", y.shape)


# STEP 5: Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


#TRAINING AND TESTING

# STEP 6: Train the Machine Learning model

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression(max_iter=1000)

# Train the model using training data
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

#CHECK ACCURACY

# STEP 7: Evaluate the model

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Predict survival for test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation Results")
print("------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#PREDICTION OF SURVIVAL

# STEP 8: Predict survival for a new passenger

# Example passenger details
new_passenger = pd.DataFrame([{
    "Pclass": 1,
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 80,
    "Sex_male": 0,
    "Embarked_Q": 0,
    "Embarked_S": 0
}])

# Predict survival
prediction = model.predict(new_passenger)

print("\nNew Passenger Prediction:")

if prediction[0] == 1:
    print("The passenger is predicted to SURVIVE.")
else:
    print("The passenger is predicted NOT TO SURVIVE.")


#SAVING THE PROJECT

# STEP 9: Save the trained model

import joblib

joblib.dump(model, "titanic_model.pkl")

print("\nModel saved successfully!")
