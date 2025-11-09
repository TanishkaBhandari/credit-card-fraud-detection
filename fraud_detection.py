# fraud_detection.py - Train model with ONLY 6 columns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


print("🔄 Loading dataset...")
# Load dataset
df = pd.read_csv("creditcard.csv")

# Use only 6 columns for demo 
cols = ['Time', 'V1', 'V2', 'V3', 'V4', 'Amount', 'Class']
df = df[cols]
print(f"✅ Using only {len(cols)-1} features: {cols[:-1]}")

# Use 10% of data for fast training
df = df.sample(frac=0.1, random_state=42)
print(f"✅ Using 10% of data: {len(df)} rows")

# Split features & target
X = df.drop('Class', axis=1)
y = df['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle imbalance
print("⚖️ Balancing dataset...")
ros = RandomOverSampler(random_state=0)
X_res, y_res = ros.fit_resample(X_train, y_train)

# Scale data
print("📊 Scaling data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_res)
X_test_scaled = scaler.transform(X_test)

# Train model 
print("🤖 Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_res)

# Test accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# Save model & scaler
joblib.dump(model, "fraud_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n🎉 SUCCESS! Model and scaler saved as:")
print("   - fraud_model.pkl")
print("   - scaler.pkl")
print("\nNow run: python app.py")


# Calculate performance metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("\n📊 Model Performance Metrics:")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Display detailed classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix heatmap
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
