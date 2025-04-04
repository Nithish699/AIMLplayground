import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

# Dummy data: [Size in sqft, Number of bedrooms]
X = np.array([
    [1000, 2], 
    [1500, 3], 
    [2000, 3], 
    [2500, 4], 
    [3000, 4]
])
y = np.array([300000, 450000, 500000, 650000, 700000])  # Prices in $

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, 'house_price_model.pkl')
print("Model saved as 'house_price_model.pkl'")
