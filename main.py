import numpy as np
import pandas as pd
import RATEC

# Load data
data = pd.read_csv("SCADA.csv")

# Features
features = [
    "Generator temperature",
    "Bearing temperature",
    "Vibration (g)",
    "Power (kw)"
]

X_train = data[features].values

# IMPORTANT:
# Use Fault Type, because RATEC will then work with class names
y_train = data["Fault Type"].values

# Risk weights must use EXACTLY the same labels as y_train
risk_weights = {
    "Normal": 1.0,
    "Bearing Fault": 2.0,
    "Overheating Fault": 3.0,
    "Electrical Fault": 4.0
}

# Create classifier
classifier = RATEC.RATEC(
    risk_weights=risk_weights,
    rho=0.6
)

# Train
classifier.fit(
    X_train,
    y_train
)

# New real-time observation
new_observation = [
    94.5,    # Generator temperature
    89.2,    # Bearing temperature
    0.82,    # Vibration
    1250.0   # Power
]

# Predict
prediction, scores = classifier.predict_one(
    new_observation
)

print("Predicted class:", prediction)
print("Risk-aware scores:", scores)