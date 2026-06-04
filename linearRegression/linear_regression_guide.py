"""
===================================================================================
                    LINEAR REGRESSION - COMPLETE GUIDE
===================================================================================

LINEAR REGRESSION SIMPLIFIED:
Imagine you're trying to predict a house's price based on its size.
- You have data of past houses (size and price)
- Linear regression finds the BEST STRAIGHT LINE that fits this data
- Then you can use this line to predict new prices!

Think of it like finding a relationship: "The bigger the house, the more it costs"

KEY CONCEPTS:
1. INPUT (Features) → size of house
2. OUTPUT (Target) → price of house
3. LINE EQUATION → y = mx + b (m=slope, b=intercept)
4. GOAL → Find the best m and b values
===================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: CREATE SAMPLE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: CREATING SAMPLE DATA")
print("=" * 80)

# Let's create data: House sizes (in sq ft) and their prices (in thousands)
np.random.seed(42)
house_sizes = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500])
prices = np.array([200, 250, 300, 350, 400, 450, 500, 550, 600, 650]) + np.random.normal(0, 20, 10)

# Put into a nice dataframe
data = pd.DataFrame({
    'House_Size_sqft': house_sizes,
    'Price_thousands': prices
})

print("\nOur Data (First 5 rows):")
print(data.head())
print(f"\nTotal houses: {len(data)}")
print(f"Size range: {data['House_Size_sqft'].min()} - {data['House_Size_sqft'].max()} sq ft")
print(f"Price range: ${data['Price_thousands'].min():.2f}k - ${data['Price_thousands'].max():.2f}k")


# ============================================================================
# STEP 2: PREPARE DATA FOR TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPARE DATA FOR TRAINING")
print("=" * 80)

# Separate input (X) and output (y)
X = data[['House_Size_sqft']].values  # INPUT: What we use to predict
y = data['Price_thousands'].values     # OUTPUT: What we want to predict

print(f"\nInput (X) shape: {X.shape}")  # (10, 1) = 10 samples, 1 feature
print(f"Output (y) shape: {y.shape}")  # (10,) = 10 samples

# Split into training and testing data
# Training data: 80% - used to train the model
# Testing data: 20% - used to check if model works on new data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")


# ============================================================================
# STEP 3: CREATE AND TRAIN THE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: CREATE AND TRAIN THE MODEL")
print("=" * 80)

# Create a Linear Regression model
model = LinearRegression()

# Train the model using training data
# This is where the magic happens! The model finds the best line.
model.fit(X_train, y_train)

print("\n✓ Model trained successfully!")
print(f"\nModel Equation: Price = {model.coef_[0]:.4f} × Size + {model.intercept_:.4f}")
print("\nTranslation:")
print(f"  - For every 1 sq ft increase, price increases by ${model.coef_[0]:.4f}k")
print(f"  - Base price (intercept): ${model.intercept_:.4f}k")


# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MAKE PREDICTIONS")
print("=" * 80)

# Predict on training data
y_train_pred = model.predict(X_train)

# Predict on testing data
y_test_pred = model.predict(X_test)

print("\nPredictions on TEST data:")
for actual, predicted in zip(y_test, y_test_pred):
    error = abs(actual - predicted)
    print(f"  Actual: ${actual:.2f}k | Predicted: ${predicted:.2f}k | Error: ${error:.2f}k")

# Predict for a NEW house we don't have data for
new_house_size = np.array([[3250]])  # 3250 sq ft
predicted_price = model.predict(new_house_size)
print(f"\n🏠 NEW PREDICTION:")
print(f"  House size: 3250 sq ft")
print(f"  Predicted price: ${predicted_price[0]:.2f}k")


# ============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: EVALUATE MODEL PERFORMANCE")
print("=" * 80)

# Calculate different performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_rmse = np.sqrt(train_mse)  # Root Mean Squared Error (in thousands $)
test_rmse = np.sqrt(test_mse)

train_mae = mean_absolute_error(y_train, y_train_pred)  # Mean Absolute Error
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)  # R² Score (closer to 1 is better)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 PERFORMANCE METRICS EXPLAINED:\n")

print("1. RMSE (Root Mean Squared Error)")
print("   - Average prediction error in thousands of dollars")
print("   - Lower is better")
print(f"   - Training RMSE: ${train_rmse:.2f}k")
print(f"   - Testing RMSE:  ${test_rmse:.2f}k")

print("\n2. MAE (Mean Absolute Error)")
print("   - Average absolute difference between actual and predicted")
print("   - Lower is better")
print(f"   - Training MAE: ${train_mae:.2f}k")
print(f"   - Testing MAE:  ${test_mae:.2f}k")

print("\n3. R² Score (Coefficient of Determination)")
print("   - How well the model explains the data (0 to 1)")
print("   - 1.0 = Perfect prediction")
print("   - 0.0 = Model is useless")
print("   - >0.7 = Generally good")
print(f"   - Training R²: {train_r2:.4f}")
print(f"   - Testing R²:  {test_r2:.4f}")

if test_r2 > 0.7:
    print("\n✅ Good model! The model fits the data well.")
elif test_r2 > 0.5:
    print("\n⚠️  Moderate model. Could be better.")
else:
    print("\n❌ Poor model. Predictions might not be reliable.")


# ============================================================================
# STEP 6: VISUALIZE THE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: CREATING VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression - Complete Analysis', fontsize=16, fontweight='bold')

# Plot 1: Regression Line
ax1 = axes[0, 0]
ax1.scatter(X_train, y_train, color='blue', s=100, label='Training Data', alpha=0.6)
ax1.scatter(X_test, y_test, color='red', s=100, label='Testing Data', alpha=0.6)

# Plot the regression line
X_line = np.array([[X.min()], [X.max()]])
y_line = model.predict(X_line)
ax1.plot(X_line, y_line, color='green', linewidth=2.5, label='Regression Line')

# Highlight the new prediction
ax1.scatter(new_house_size, predicted_price, color='orange', s=150, marker='*', 
            label='New Prediction', edgecolors='black', linewidth=2)

ax1.set_xlabel('House Size (sq ft)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Price ($thousands)', fontsize=11, fontweight='bold')
ax1.set_title('Regression Line Fit', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)


# Plot 2: Residuals (Errors)
ax2 = axes[0, 1]
residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

ax2.scatter(y_train_pred, residuals_train, color='blue', s=100, label='Training Residuals', alpha=0.6)
ax2.scatter(y_test_pred, residuals_test, color='red', s=100, label='Testing Residuals', alpha=0.6)
ax2.axhline(y=0, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Price ($thousands)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (Error)', fontsize=11, fontweight='bold')
ax2.set_title('Residuals Plot (Should be random around 0)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)


# Plot 3: Actual vs Predicted
ax3 = axes[1, 0]
all_actual = np.concatenate([y_train, y_test])
all_predicted = np.concatenate([y_train_pred, y_test_pred])

ax3.scatter(all_actual, all_predicted, color='purple', s=100, alpha=0.6)
# Perfect prediction line (y=x)
min_val = min(all_actual.min(), all_predicted.min())
max_val = max(all_actual.max(), all_predicted.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Price ($thousands)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Predicted Price ($thousands)', fontsize=11, fontweight='bold')
ax3.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)


# Plot 4: Performance Metrics
ax4 = axes[1, 1]
ax4.axis('off')

metrics_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*40}

Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}

TRAINING SET:
  • RMSE: ${train_rmse:.2f}k
  • MAE:  ${train_mae:.2f}k
  • R²:   {train_r2:.4f}

TESTING SET:
  • RMSE: ${test_rmse:.2f}k
  • MAE:  ${test_mae:.2f}k
  • R²:   {test_r2:.4f}

INTERPRETATION:
  • Slope: ${model.coef_[0]:.2f}k per sq ft
  • Intercept: ${model.intercept_:.2f}k (base)
  
PREDICTION EXAMPLE:
  3250 sq ft house → ${predicted_price[0]:.2f}k
"""

ax4.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('c:\\Users\\asus\\OneDrive\\Desktop\\ML\\linear_regression_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'linear_regression_analysis.png'")
plt.show()

print("\n" + "=" * 80)
print("✅ LINEAR REGRESSION ANALYSIS COMPLETE!")
print("=" * 80)
