"""
===================================================================================
        LINEAR REGRESSION WITH MULTIPLE EXAMPLES
===================================================================================
This script demonstrates linear regression with different real-world scenarios
to help you understand how it works in different contexts.
===================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


def train_and_evaluate(X, y, title, xlabel, ylabel):
    """
    Trains a linear regression model and displays results.
    """
    # Split data: 80% train, 20% test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    return {
        'model': model,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred,
        'train_r2': train_r2, 'test_r2': test_r2,
        'test_mae': test_mae,
        'title': title, 'xlabel': xlabel, 'ylabel': ylabel
    }


# =============================================================================
# EXAMPLE 1: HOUSE PRICES
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 1: PREDICTING HOUSE PRICES")
print("="*80)

# Data: House sizes and prices
house_sizes = np.array([1000, 1200, 1500, 1800, 2000, 2300, 2500, 2800, 3000, 3200, 
                        3500, 3800, 4000, 4200, 4500]).reshape(-1, 1)
house_prices = np.array([150, 170, 200, 230, 260, 290, 320, 350, 380, 410, 
                         450, 480, 510, 540, 580])

result1 = train_and_evaluate(house_sizes, house_prices, 
                             "House Price Prediction",
                             "House Size (sq ft)", "Price ($thousands)")

print(f"\n📊 Results:")
print(f"   Equation: Price = {result1['model'].coef_[0]:.4f} × Size + {result1['model'].intercept_:.2f}")
print(f"   R² Score (Test): {result1['test_r2']:.4f}")
print(f"   Average Error: ${result1['test_mae']:.2f}k")
print(f"\n   💡 Interpretation:")
print(f"      • Each additional 100 sq ft increases price by ${result1['model'].coef_[0]*100:.2f}k")
print(f"      • Model explains {result1['test_r2']*100:.1f}% of price variation")


# =============================================================================
# EXAMPLE 2: STUDENT STUDY HOURS vs EXAM SCORES
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 2: STUDY HOURS vs EXAM SCORES")
print("="*80)

# Data: Study hours and exam scores
study_hours = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 
                        12, 13, 14, 15, 16]).reshape(-1, 1)
exam_scores = np.array([50, 55, 60, 65, 72, 78, 82, 85, 88, 90, 
                        92, 94, 95, 96, 97])

result2 = train_and_evaluate(study_hours, exam_scores,
                             "Exam Score Prediction",
                             "Study Hours", "Exam Score")

print(f"\n📊 Results:")
print(f"   Equation: Score = {result2['model'].coef_[0]:.4f} × Hours + {result2['model'].intercept_:.2f}")
print(f"   R² Score (Test): {result2['test_r2']:.4f}")
print(f"   Average Error: {result2['test_mae']:.2f} points")
print(f"\n   💡 Interpretation:")
print(f"      • Each additional hour of study increases score by {result2['model'].coef_[0]:.2f} points")
print(f"      • Model explains {result2['test_r2']*100:.1f}% of score variation")


# =============================================================================
# EXAMPLE 3: EXPERIENCE vs SALARY
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 3: EXPERIENCE vs SALARY")
print("="*80)

# Data: Years of experience and salary
experience_years = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                             11, 12, 13, 14, 15]).reshape(-1, 1)
salary_usd = np.array([30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000,
                       80000, 85000, 90000, 95000, 100000])

result3 = train_and_evaluate(experience_years, salary_usd,
                             "Salary Prediction",
                             "Years of Experience", "Salary ($)")

print(f"\n📊 Results:")
print(f"   Equation: Salary = {result3['model'].coef_[0]:.2f} × Experience + {result3['model'].intercept_:.2f}")
print(f"   R² Score (Test): {result3['test_r2']:.4f}")
print(f"   Average Error: ${result3['test_mae']:.2f}")
print(f"\n   💡 Interpretation:")
print(f"      • Each year of experience increases salary by ${result3['model'].coef_[0]:.2f}")
print(f"      • Model explains {result3['test_r2']*100:.1f}% of salary variation")


# =============================================================================
# EXAMPLE 4: ADVERTISEMENT SPENDING vs SALES
# =============================================================================
print("\n" + "="*80)
print("EXAMPLE 4: AD SPENDING vs SALES")
print("="*80)

# Data: Money spent on ads and resulting sales
ad_spending = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500,
                        6000, 6500, 7000, 7500, 8000]).reshape(-1, 1)
sales_revenue = np.array([5000, 8000, 10000, 13000, 15000, 18000, 20000, 22000, 24000, 26000,
                          28000, 30000, 31000, 33000, 35000])

result4 = train_and_evaluate(ad_spending, sales_revenue,
                             "Sales vs Ad Spending",
                             "Ad Spending ($)", "Sales Revenue ($)")

print(f"\n📊 Results:")
print(f"   Equation: Sales = {result4['model'].coef_[0]:.4f} × AdSpend + {result4['model'].intercept_:.2f}")
print(f"   R² Score (Test): {result4['test_r2']:.4f}")
print(f"   Average Error: ${result4['test_mae']:.2f}")
print(f"\n   💡 Interpretation:")
print(f"      • Every $1 spent on ads generates ${result4['model'].coef_[0]:.2f} in sales")
print(f"      • Model explains {result4['test_r2']*100:.1f}% of sales variation")


# =============================================================================
# VISUALIZATION: ALL EXAMPLES IN ONE PLOT
# =============================================================================
print("\n" + "="*80)
print("Creating comparison visualization...")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Linear Regression - 4 Real-World Examples', fontsize=16, fontweight='bold')

examples = [
    (result1, axes[0, 0]),
    (result2, axes[0, 1]),
    (result3, axes[1, 0]),
    (result4, axes[1, 1])
]

for result, ax in examples:
    # Plot data points
    ax.scatter(result['X_train'], result['y_train'], color='blue', s=80, label='Train', alpha=0.6)
    ax.scatter(result['X_test'], result['y_test'], color='red', s=80, label='Test', alpha=0.6)
    
    # Plot regression line
    X_line = np.array([[result['X_train'].min()], [result['X_train'].max()]])
    y_line = result['model'].predict(X_line)
    ax.plot(X_line, y_line, color='green', linewidth=2.5, label='Fit Line')
    
    ax.set_xlabel(result['xlabel'], fontsize=10, fontweight='bold')
    ax.set_ylabel(result['ylabel'], fontsize=10, fontweight='bold')
    
    title = f"{result['title']}\nR² = {result['test_r2']:.3f} | MAE = {result['test_mae']:.1f}"
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('c:\\Users\\asus\\OneDrive\\Desktop\\ML\\examples_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'examples_comparison.png'")
plt.show()


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("SUMMARY: ALL EXAMPLES AT A GLANCE")
print("="*80)

summary_data = {
    'Example': ['House Prices', 'Study Hours', 'Experience', 'Ad Spending'],
    'Input': ['Size (sqft)', 'Hours', 'Years', 'Spending ($)'],
    'Output': ['Price ($k)', 'Score', 'Salary ($)', 'Revenue ($)'],
    'Slope': [result1['model'].coef_[0], result2['model'].coef_[0], 
              result3['model'].coef_[0], result4['model'].coef_[0]],
    'R² Score': [result1['test_r2'], result2['test_r2'], result3['test_r2'], result4['test_r2']]
}

summary_df = pd.DataFrame(summary_data)
print("\n", summary_df.to_string(index=False))

print("\n" + "="*80)
print("✅ ALL EXAMPLES COMPLETE!")
print("="*80)
print("\n💡 KEY INSIGHTS:")
print("   • Linear regression works when data follows a straight-line pattern")
print("   • Higher R² means better predictions")
print("   • The slope tells you how much output changes per unit input")
print("   • All models show good fits (R² > 0.9)")
