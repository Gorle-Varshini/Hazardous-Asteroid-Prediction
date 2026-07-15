from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

# Dummy scaler trained on minimal data
scaler = StandardScaler()
scaler.fit(np.array([[0.0],[1.0]]))

# Dummy classifier that always predicts 0
model = DummyClassifier(strategy='constant', constant=0)
model.fit([[0,0,0,0,0,0,0]], [0])

joblib.dump(model, 'asteroid_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print('Created asteroid_model.pkl and scaler.pkl')
