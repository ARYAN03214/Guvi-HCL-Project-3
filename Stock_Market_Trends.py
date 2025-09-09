# Predictive Modeling of Stock Market Trends

# 1. Problem Definition & Understanding
# Objective: Predict next day stock closing price trend (Up/Down) using historical data.
# Stock market data is time-series data, and trend prediction is a classification problem.

# 2. Data Collection & Cleaning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import yfinance as yf

# Download historical stock data (e.g., Apple) from Yahoo Finance
df = yf.download('AAPL', start='2019-01-01', end='2023-01-01')

# Clean data: check for nulls and fill/drop them if any
print("Null values check:")
print(df.isnull().sum())
df.dropna(inplace=True)  # No nulls in this data generally, but drop just in case

# 3. Data Analysis & Charts
# Display first few rows of data
print(df.head())

# Plot closing price history
plt.figure(figsize=(14,7))
plt.plot(df['Close'], label='Close Price History')
plt.title('Apple Stock Closing Price (Historical)')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.show()

# Plot daily returns distribution to understand volatility
df['Daily_Return'] = df['Close'].pct_change()
plt.figure(figsize=(10,5))
sns.histplot(df['Daily_Return'].dropna(), bins=50)
plt.title("Distribution of Daily Returns")
plt.show()

# 4. Feature Engineering & Model Building
# Define target variable as next day trend: 1 (price up), 0 (price down or no change)
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Use features: Open, High, Low, Close, Volume, and Moving Averages
df['MA5'] = df['Close'].rolling(window=5).mean()
df['MA10'] = df['Close'].rolling(window=10).mean()
df = df.dropna()  # Drop rows with NaN values due to rolling

# Features and target
features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA5', 'MA10']
X = df[features]
y = df['Target']

# Split dataset into training and testing sets (70:30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Building: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Model Checking
y_pred = model.predict(X_test_scaled)

# Confusion matrix and classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Results & Insights
# Calculate accuracy & precision to evaluate model performance
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Predicting Stock Trend')
plt.show()

# Insights: Moving averages and volume seem important for trend prediction.
# The model accuracy shows that stock price movement prediction is challenging but feasible.

# 7. Report & Documentation
# Note: In production, this would include detailed documentation alongside the code.
# Comments in code explain each step.
# Further improvement: Use more advanced models like LSTM, XGBoost, and more features (news sentiment).

