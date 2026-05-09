3.2 Random Forest --
print("\n[3.2] Random Forest...")
rf = RandomForestRegressor(
    n_estimators=500, max_depth=25, min_samples_split=5,
    min_samples_leaf=2, max_features='sqrt',
    n_jobs=-1, random_state=42)
rf.fit(X_train_tree, y_train)
evaluate("Random Forest", y_test, rf.predict(X_test_tree))

# -- 3.3 XGBoost (Optuna-tuned) --
print("\n[3.3] XGBoost (Optuna-tuned)...")
xgb_model = xgb.XGBRegressor(
    n_estimators=837,
    max_depth=7,
    learning_rate=0.021637453876444356,
    subsample=0.7891972163208942,
    colsample_bytree=0.7188139055430017,
    reg_alpha=3.4223384065280196,
    reg_lambda=0.0230615093634716,
    min_child_weight=9,
    n_jobs=-1, random_state=42, verbosity=0, tree_method='hist')
xgb_model.fit(X_train_tree, y_train,
              eval_set=[(X_test_tree, y_test)], verbose=False)
evaluate("XGBoost", y_test, xgb_model.predict(X_test_tree))

# -- 3.4 LightGBM (Optuna-tuned) --
print("\n[3.4] LightGBM (Optuna-tuned)...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=536,
    max_depth=10,
    learning_rate=0.029494141218233967,
    subsample=0.8803196307012858,
    colsample_bytree=0.6004543702566304,
    num_leaves=91,
    min_child_samples=30,
    reg_alpha=0.0664416304461952,
    reg_lambda=0.06960907687505945,
    n_jobs=-1, random_state=42, verbose=-1)
lgb_model.fit(X_train_tree, y_train,
              eval_set=[(X_test_tree, y_test)])
evaluate("LightGBM", y_test, lgb_model.predict(X_test_tree))

# -- 3.5 Stacking Ensemble --
print("\n[3.5] Stacking Ensemble...")
estimators = [
    ('rf', RandomForestRegressor(
        n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)),
    ('xgb', xgb.XGBRegressor(
        n_estimators=837, max_depth=7,
        learning_rate=0.021637453876444356,
        subsample=0.7891972163208942,
        colsample_bytree=0.7188139055430017,
        reg_alpha=3.4223384065280196,
        reg_lambda=0.0230615093634716,
        min_child_weight=9,
        n_jobs=-1, random_state=42, verbosity=0)),
    ('lgb', lgb.LGBMRegressor(
        n_estimators=536, max_depth=10,
        learning_rate=0.029494141218233967,
        subsample=0.8803196307012858,
        colsample_bytree=0.6004543702566304,
        num_leaves=91, min_child_samples=30,
        reg_alpha=0.0664416304461952,
        reg_lambda=0.06960907687505945,
        n_jobs=-1, random_state=42, verbose=-1)),
]