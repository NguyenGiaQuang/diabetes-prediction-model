import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
import joblib

def train_tree():
    data = pd.read_csv("diabetes.csv")

    le = LabelEncoder()
    data["gender"] = le.fit_transform(data["gender"])
    data["smoking_history"] = le.fit_transform(data["smoking_history"])

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, "tree.pkl")
    joblib.dump(scaler, "scaler_tree.pkl")

if __name__ == "__main__":
    train_tree()
