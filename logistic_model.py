import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic():
    data = pd.read_csv("diabetes.csv")

    # Encode categorical
    le = LabelEncoder()
    data["gender"] = le.fit_transform(data["gender"])
    data["smoking_history"] = le.fit_transform(data["smoking_history"])

    X = data.drop("diabetes", axis=1)
    y = data["diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Lưu model và scaler
    joblib.dump(model, "logistic.pkl")
    joblib.dump(scaler, "scaler_logistic.pkl")

if __name__ == "__main__":
    train_logistic()
