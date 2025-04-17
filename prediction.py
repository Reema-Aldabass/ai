
import joblib
import pandas as pd

# حمل النماذج
model = joblib.load('model.joblib')
scaler = joblib.load('scalar.joblib')

X_test = pd.read_csv('X_test.csv')
X_test_scaled = scaler.transform(X_test)

y_pred = model.predict(X_test_scaled)

print("Predictions:")
print(y_pred)


try:
    y_test = pd.read_csv('y_test.csv')
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc * 100:.2f}%")
except FileNotFoundError:
    print("y_test.csv not found. Skipping accuracy.")
