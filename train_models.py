import pandas as pd
import os, joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Đọc dữ liệu
data = pd.read_csv("diabetes.csv")

# Encode cột categorical
cat_cols = ["gender", "smoking_history"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# X/y
X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Các model
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "tree": DecisionTreeClassifier(max_depth=5),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "svm": SVC(kernel="rbf", probability=True),
}

# Tạo thư mục models
if not os.path.exists("models"):
    os.makedirs("models")

# Train và lưu từng model
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    joblib.dump(model, f"models/{name}.pkl")

# Lưu scaler + encoders
joblib.dump(scaler, "models/scaler.pkl")
for col, le in encoders.items():
    joblib.dump(le, f"models/{col}_encoder.pkl")

print("✅ Đã train xong và lưu model + scaler + encoders")
