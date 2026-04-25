from hyper_parameter_tuning import *

# ===========================================================
# PART 4: RESULTS
# ===========================================================
print("\n\n" + "=" * 70)
print("  PART 4: RESULTS")
print("=" * 70)

res_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
print("\n" + res_df.to_string())

# v3 baseline
v3_r2 = 0.37398
best_name = res_df.index[0]
best_r2 = res_df.loc[best_name, 'R2']
improvement = ((best_r2 - v3_r2) / v3_r2) * 100

print(f"\n  v3 best R2 (XGBoost):  {v3_r2:.5f}")
print(f"  v5 best R2 ({best_name}): {best_r2:.5f}")
print(f"  Improvement: {improvement:+.1f}%")

# Model comparison chart
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, metric in enumerate(['R2', 'MAE', 'RMSE']):
    ax = axes[i]
    vals = res_df[metric]
    best = vals.max() if metric == 'R2' else vals.min()
    colors = ['#27ae60' if v == best else '#bdc3c7' for v in vals]
    ax.barh(vals.index, vals.values, color=colors)
    ax.set_xlabel(metric)
    ax.set_title(metric)
plt.suptitle('Model Comparison v5 - Predicting Score', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('visuales/model_comparison_v5.png', dpi=150)
plt.show()

# Feature importance from best tree model
# Feature importance from best tree model
print("\nExtracting feature importance...")

# ALWAYS use model input features (safe source)
model_features = X_train.columns

if 'XGBoost' in best_name:
    imp_vals = xgb_model.feature_importances_
elif best_name == 'LightGBM':
    imp_vals = lgb_model.feature_importances_
elif best_name == 'Random Forest':
    imp_vals = rf.feature_importances_
else:
    imp_vals = np.abs(lr.coef_)

# Safety check
print(f"Features: {len(model_features)}, Importances: {len(imp_vals)}")

# Align safely (this prevents crash)
min_len = min(len(model_features), len(imp_vals))

imp = pd.DataFrame({
    'feature': model_features[:min_len],
    'importance': imp_vals[:min_len]
})

imp = imp.sort_values('importance', ascending=False)

print(f"\n  {best_name} Top 15 Feature Importances:")
print(imp.head(15).to_string(index=False))

plt.figure(figsize=(10, 8))
top = imp.head(20).set_index('feature')['importance']
plt.barh(top.index[::-1], top.values[::-1], color='#3498db')
plt.xlabel('Importance')
plt.title(f'{best_name} - Top 20 Features')
plt.tight_layout()
plt.savefig('visuales/feature_importance_v5.png', dpi=150)
plt.show()

# Actual vs Predicted
if 'Stacking' in best_name:
    y_best = stack.predict(X_test)
elif 'XGBoost' in best_name:
    y_best = xgb_model.predict(X_test)
elif best_name == 'LightGBM':
    y_best = lgb_model.predict(X_test)
elif best_name == 'Random Forest':
    y_best = rf.predict(X_test)
else:
    y_best = lr.predict(X_test)

plt.figure(figsize=(8, 8))
s = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
plt.scatter(y_test.values[s], y_best[s], alpha=0.15, s=5, color='#3498db')
plt.plot([0, 100], [0, 100], 'r--', linewidth=1, label='Perfect')
plt.xlabel('Actual Score')
plt.ylabel('Predicted Score')
plt.title(f'{best_name}: Actual vs Predicted (R2={best_r2:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('visuales/actual_vs_predicted_v5.png', dpi=150)
plt.show()

# Final
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"""
  Target:     score
  Rows:       {len(df)}
  Features:   {len(feature_cols)}

  New in v5:
    + Interaction features (clicks*timing, edu*clicks, etc.)
    + Log-transforms for skewed VLE features
    + Prior score std deviation
    + abs_days_from_deadline
    + Deeper trees (max_depth=25 RF, 10-12 XGB/LGB)
    + More estimators (500-700)

  Model Results:
""")
print(res_df.to_string())
print(f"\n  Best: {best_name} (R2={best_r2:.5f}, improvement: {improvement:+.1f}% over v3)")
print("  Done!")