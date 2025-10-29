from flask import Flask, render_template, request, send_file
import subprocess, joblib, os, io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# Nếu models chưa có thì chạy train_models.py
if not os.path.exists("models") or not all(
    os.path.exists(f"models/{m}.pkl") for m in ["logistic", "tree", "knn", "svm"]
):
    print("⚠️ Chưa có model, đang huấn luyện...")
    subprocess.run(["python", "train_models.py"], check=True)
    print("✅ Huấn luyện xong, tiếp tục load model...")

# Load models + scaler + encoders
models = {
    "Logistic Regression": joblib.load("models/logistic.pkl"),
    "Decision Tree": joblib.load("models/tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
}
scaler = joblib.load("models/scaler.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")
smoking_encoder = joblib.load("models/smoking_history_encoder.pkl")

# Đọc dữ liệu để đánh giá
data = pd.read_csv("diabetes.csv")
for col in ["gender", "smoking_history"]:
    le = joblib.load(f"models/{col}_encoder.pkl")
    data[col] = le.transform(data[col])

X = data.drop("diabetes", axis=1)
y = data["diabetes"]

# Metrics
model_scores = {}
X_scaled = scaler.transform(X)
for name, model in models.items():
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    model_scores[name] = {"ACC": acc, "MAE": mae, "MSE": mse, "R2": r2}


@app.route("/", methods=["GET", "POST"])
def index():
    probs = {}
    if request.method == "POST":
        # Lấy input
        gender = request.form["gender"]
        age = float(request.form["age"])
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        smoking = request.form["smoking_history"]
        bmi = float(request.form["bmi"])
        HbA1c_level = float(request.form["HbA1c_level"])
        glucose = float(request.form["blood_glucose_level"])

        # Encode categorical
        gender_val = gender_encoder.transform([gender])[0]
        smoking_val = smoking_encoder.transform([smoking])[0]

        row = np.array([[gender_val, age, hypertension, heart_disease,
                         smoking_val, bmi, HbA1c_level, glucose]])

        row_scaled = scaler.transform(row)

        for name, model in models.items():
            prob = model.predict_proba(row_scaled)[0][1]
            pred = model.predict(row_scaled)[0]
            probs[name] = (pred, prob)

    return render_template("index.html", results=probs, metrics=model_scores)


@app.route("/accuracy.png")
def accuracy_chart():
    fig, ax = plt.subplots()
    accs = {name: vals["ACC"] for name, vals in model_scores.items()}
    ax.bar(accs.keys(), [v * 100 for v in accs.values()], color="orange")
    plt.ylabel("Accuracy (%)")
    plt.title("So sánh Accuracy giữa các mô hình")
    plt.ylim(0, 100)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/chart.png")
def chart():
    """Vẽ biểu đồ xác suất cho input giả định"""
    sample = np.array([[gender_encoder.transform(["Male"])[0],
                        40, 0, 0,
                        smoking_encoder.transform(["never"])[0],
                        25, 5.5, 120]])
    sample_scaled = scaler.transform(sample)

    results = {}
    for name, model in models.items():
        prob = model.predict_proba(sample_scaled)[0][1]
        results[name] = prob

    fig, ax = plt.subplots()
    ax.bar(results.keys(), results.values(), color="skyblue")
    plt.ylabel("Xác suất mắc bệnh")
    plt.title("So sánh xác suất dự đoán (sample input)")
    plt.ylim(0, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
