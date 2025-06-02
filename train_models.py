import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report

# Load dataset
data = pd.read_csv('C:/Users/prath/OneDrive/Documents/Desktop/student mangement system/students.csv')
data.dropna(inplace=True)

# Create 'At_Risk' binary column
data['At_Risk'] = data['Performance Label'].apply(lambda x: 1 if x == 'Low' else 0)

features = ['Sem1', 'Sem2', 'Sem3', 'Attendance', 'Activities']
X = data[features]
y_reg = data['Final Exam Score']
y_cls = data['At_Risk']

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create models directory
os.makedirs('models', exist_ok=True)

# Save scaler
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print("Regression RMSE:", np.sqrt(mean_squared_error(y_test, reg_model.predict(X_test))))
with open('models/performance_model.pkl', 'wb') as f:
    pickle.dump(reg_model, f)

# Classification Model
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled, y_cls, test_size=0.2, random_state=42)
cls_model = LogisticRegression(max_iter=1000)
cls_model.fit(X_train_cls, y_train_cls)
print("Classification Report:\n", classification_report(y_test_cls, cls_model.predict(X_test_cls)))
with open('models/at_risk_model.pkl', 'wb') as f:
    pickle.dump(cls_model, f)

print("\n✅ Models trained and saved successfully.")
