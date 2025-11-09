# predict_fraud.py
import pandas as pd
import joblib

# Step 1: Load saved model and scaler
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
print("✅ Model and scaler loaded successfully!\n")

# Step 2: Example transaction input 
sample_data = {
    'Time': [100000],
    'V1': [-1.359807],
    'V2': [-0.072781],
    'V3': [2.536347],
    'V4': [1.378155],
    'Amount': [149.62]
}

# Convert to DataFrame
df = pd.DataFrame(sample_data)

# Step 3: Scale the data
df_scaled = scaler.transform(df)

# Step 4: Predict
prediction = model.predict(df_scaled)[0]

# Step 5: Show result
if prediction == 1:
    print("🚨 FRAUD DETECTED!")
else:
    print("✅ Transaction is LEGITIMATE.")
