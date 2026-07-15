from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Train scaler with 7 features
X = np.array([
    [0.02, 0.05, 25000, 45000000, 0, 0, 22],
    [0.10, 0.20, 50000, 10000000, 0, 0, 18]
])

scaler = StandardScaler()
scaler.fit(X)

# Dummy model
y = np.array([0, 1])

model = DummyClassifier(strategy="most_frequent")
model.fit(X, y)

joblib.dump(model, "asteroid_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Artifacts created successfully!")