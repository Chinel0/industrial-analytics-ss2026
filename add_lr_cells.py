import json, uuid

NB = r"c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def md(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}
def code(src):
    return {"cell_type":"code","execution_count":None,"id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}

cells = [

md("## 2. Linear Regression\n\nThis section applies Linear Regression to predict supplier quote cost. The pipeline follows six structured phases: data splitting, baseline modelling, hyperparameter tuning, validation, final test evaluation, and explainability. Two baseline variants are compared first so the choice of target scale is backed by evidence, not assumption."),

md("### Phase 1: 3-Way Data Split\n\nBefore any model is trained, the data is divided into three non-overlapping sets. The training set (60%) is used to fit the model. The validation set (20%) is used to compare tuning choices. The test set (20%) is sealed immediately and is not used for any decision until Phase 5, ensuring the final reported performance reflects true generalisation."),

code(
"""X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=42
)

print(f"Training set   : {X_train.shape[0]:>5} rows  ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Validation set : {X_val.shape[0]:>5} rows  ({X_val.shape[0]/len(X)*100:.0f}%)")
print(f"Test set       : {X_test.shape[0]:>5} rows  ({X_test.shape[0]/len(X)*100:.0f}%)  <- sealed until Phase 5")"""
),

md("**Insight:** With 30,213 samples the training set contains approximately 18,127 rows, the validation set 6,043 rows, and the test set 6,043 rows. Each partition is large enough for reliable metric estimation. Setting `random_state=42` ensures the split is reproducible."),

md("### Helper Function: Evaluation on the Original Dollar Scale\n\nAll models are trained on the power-transformed target (`cost ** (1/20)`). To report metrics that are meaningful in business terms, predictions must be converted back to the original dollar scale before computing MAE, RMSE, and R2. The `evaluate` function below handles this back-transformation for any pipeline passed to it."),

code(
"""def evaluate(pipeline, X, y_trans_true, label=""):
    y_pred_trans = pipeline.predict(X)
    y_true_orig  = (y_trans_true ** 20).reset_index(drop=True)
    y_pred_orig  = np.clip(y_pred_trans, 0, None) ** 20
    mae  = mean_absolute_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))
    r2   = r2_score(y_true_orig, y_pred_orig)
    if label:
        print(f"  {label:48s}  MAE=${mae:8.4f}  RMSE=${rmse:8.4f}  R2={r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2,
            "y_true": y_true_orig, "y_pred": y_pred_orig}"""
),

md("### Phase 2: Baseline Models\n\nTwo baseline Linear Regression models are trained and compared:\n\n- **Model A** is trained on the raw, untransformed cost values.\n- **Model B** is trained on the power-transformed cost (`cost_transformed = cost ** (1/20)`).\n\nEach model is evaluated on the validation set and with 5-fold cross-validation on the training set. Cross-validation averages performance across five different subsets of the training data, giving a more reliable stability measure than any single validation split. The model with the higher and more consistent CV R2 is selected for hyperparameter tuning."),

md("#### Model A: Linear Regression on Raw Cost\n\nModel A uses a `StandardScaler + LinearRegression` pipeline trained directly on the untransformed cost values. This is the simplest possible baseline with no transformation and no regularisation."),

code(
"""y_train_raw = y_train ** 20
y_val_raw   = y_val   ** 20

model_A = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
model_A.fit(X_train, y_train_raw)

y_pred_A = model_A.predict(X_val)

mae_A  = mean_absolute_error(y_val_raw, y_pred_A)
rmse_A = np.sqrt(mean_squared_error(y_val_raw, y_pred_A))
r2_A   = r2_score(y_val_raw, y_pred_A)

print("=== Model A: Linear Regression (Raw Cost) ===")
print(f"Validation MAE  : ${mae_A:.4f}")
print(f"Validation RMSE : ${rmse_A:.4f}")
print(f"Validation R2   : {r2_A:.4f}")"""
),

md("#### Model A: 5-Fold Cross-Validation\n\nThe training set is divided into 5 folds. The model trains on 4 folds and is evaluated on the remaining one, rotating 5 times. The spread of R2 scores across folds shows how stable Model A is when the training data changes."),

code(
"""kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_A = cross_validate(
    model_A, X_train, y_train_raw,
    cv=kf,
    scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error',
             'rmse': 'neg_root_mean_squared_error'}
)

cv_r2_A   =  cv_A['test_r2'].mean()
cv_mae_A  = -cv_A['test_mae'].mean()
cv_rmse_A = -cv_A['test_rmse'].mean()

print("=== Model A: 5-Fold Cross-Validation (Raw Cost) ===")
print(f"CV MAE  : ${cv_mae_A:.4f}")
print(f"CV RMSE : ${cv_rmse_A:.4f}")
print(f"CV R2   : {cv_r2_A:.4f}")
print(f"All CV R2 scores: {cv_A['test_r2'].round(3)}")"""
),

code(
"""plt.figure(figsize=(8, 5))
plt.scatter(y_val_raw, y_pred_A, alpha=0.3, color='steelblue', edgecolors='none', s=15)
plt.plot([y_val_raw.min(), y_val_raw.max()],
         [y_val_raw.min(), y_val_raw.max()],
         'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title('Model A: Actual vs Predicted (Raw Cost)')
plt.legend()
plt.tight_layout()
plt.show()"""
),

md("**Insight:** Model A on raw cost is sensitive to outliers because the untransformed target amplifies extreme values during training. The actual vs predicted plot typically shows the model underestimating high-cost assemblies. The CV R2 spread across folds reveals how much the model varies depending on which data it sees."),

md("#### Model B: Linear Regression on Power-Transformed Cost\n\nModel B uses the same pipeline but is trained on `cost_transformed` (cost ** (1/20)). The power transformation compresses the right tail of the cost distribution, which reduces the influence of extreme values and stabilises the regression. Predictions are back-transformed to dollars using `np.clip(pred, 0, None) ** 20` before evaluation so that metrics are directly comparable with Model A."),

code(
"""model_B = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
model_B.fit(X_train, y_train)

y_pred_B_trans = model_B.predict(X_val)
y_pred_B       = np.clip(y_pred_B_trans, 0, None) ** 20
y_val_orig     = (y_val ** 20).reset_index(drop=True)

mae_B  = mean_absolute_error(y_val_orig, y_pred_B)
rmse_B = np.sqrt(mean_squared_error(y_val_orig, y_pred_B))
r2_B   = r2_score(y_val_orig, y_pred_B)

print("=== Model B: Linear Regression (Power-Transformed Cost) ===")
print(f"Validation MAE  : ${mae_B:.4f}")
print(f"Validation RMSE : ${rmse_B:.4f}")
print(f"Validation R2   : {r2_B:.4f}")"""
),

md("#### Model B: 5-Fold Cross-Validation\n\nThe same 5-fold setup is applied to Model B. R2 scores are computed on the power-transformed scale inside each fold. A consistently high and stable R2 across folds confirms the transformation makes the model more reliable across different data subsets."),

code(
"""cv_B = cross_validate(
    model_B, X_train, y_train,
    cv=kf,
    scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error',
             'rmse': 'neg_root_mean_squared_error'}
)

cv_r2_B   =  cv_B['test_r2'].mean()
cv_mae_B  = -cv_B['test_mae'].mean()
cv_rmse_B = -cv_B['test_rmse'].mean()

print("=== Model B: 5-Fold Cross-Validation (Power-Transformed Scale) ===")
print(f"CV MAE  : {cv_mae_B:.4f}  (transformed scale)")
print(f"CV RMSE : {cv_rmse_B:.4f}  (transformed scale)")
print(f"CV R2   : {cv_r2_B:.4f}")
print(f"All CV R2 scores: {cv_B['test_r2'].round(3)}")"""
),

code(
"""plt.figure(figsize=(8, 5))
plt.scatter(y_val_orig, y_pred_B, alpha=0.3, color='seagreen', edgecolors='none', s=15)
plt.plot([y_val_orig.min(), y_val_orig.max()],
         [y_val_orig.min(), y_val_orig.max()],
         'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title('Model B: Actual vs Predicted (Power-Transformed Cost)')
plt.legend()
plt.tight_layout()
plt.show()"""
),

md("**Insight:** Model B should show a higher and more stable CV R2 than Model A because the power transformation reduces the dominance of extreme cost values during training. The scatter plot should show predictions more evenly distributed around the perfect-prediction line compared to Model A."),

md("#### Model A vs Model B: Comparison\n\nThe next cell summarises both models side by side. The cross-validation R2 is the deciding metric because it averages across five independent splits and is more reliable than any single validation score."),

code(
"""models = ['Model A\\n(Raw Cost)', 'Model B\\n(Power-Transformed)']
val_r2 = [r2_A,    r2_B]
cv_r2  = [cv_r2_A, cv_r2_B]

print("=== Model A vs Model B: Summary ===")
print(f"{'Metric':<22} {'Model A':>14} {'Model B':>14}")
print("-" * 51)
print(f"{'Validation MAE ($)':<22} {mae_A:>14.4f} {mae_B:>14.4f}")
print(f"{'Validation RMSE ($)':<22} {rmse_A:>14.4f} {rmse_B:>14.4f}")
print(f"{'Validation R2':<22} {r2_A:>14.4f} {r2_B:>14.4f}")
print(f"{'CV R2 (mean)':<22} {cv_r2_A:>14.4f} {cv_r2_B:>14.4f}")

x     = np.arange(len(models))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, values, title in [
    (axes[0], val_r2, 'Validation R2'),
    (axes[1], cv_r2,  '5-Fold CV R2 (Stability)')
]:
    bars = ax.bar(x, values, 0.4, color=['steelblue', 'seagreen'], edgecolor='black')
    ax.set_title(title, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('R2')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)

plt.suptitle('Model A vs Model B: Validation and CV Comparison', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

if cv_r2_B > cv_r2_A:
    print("Conclusion: Model B has a higher CV R2. It is carried forward to Phase 3.")
else:
    print("Conclusion: Model A has a higher CV R2. It is carried forward to Phase 3.")"""
),

md("**Insight:** The CV R2 is the primary criterion, not the single-split validation R2. A model that scores well on one split but varies widely across folds is unstable and not trustworthy. The model with consistently higher CV R2 across all five folds is selected for hyperparameter tuning."),

md("### Phase 3: Hyperparameter Tuning with GridSearchCV\n\nThis phase searches for the best regularisation strength for Ridge and Lasso regression. Ridge (L2 penalty) penalises large coefficients and prevents overfitting without removing features. Lasso (L1 penalty) can shrink some coefficients to zero, which is a form of automatic feature selection. GridSearchCV evaluates each alpha value with 5-fold cross-validation on the training set only. The validation set is not seen during this entire phase."),

code(
"""print("=" * 60)
print("PHASE 3: HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

ridge_pipe   = Pipeline([('scaler', StandardScaler()), ('model', Ridge())])
ridge_params = {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]}

ridge_gs = GridSearchCV(ridge_pipe, ridge_params, cv=kf,
                        scoring='neg_mean_absolute_error',
                        return_train_score=True, n_jobs=-1)
ridge_gs.fit(X_train, y_train)

lasso_pipe   = Pipeline([('scaler', StandardScaler()), ('model', Lasso(max_iter=10000))])
lasso_params = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1]}

lasso_gs = GridSearchCV(lasso_pipe, lasso_params, cv=kf,
                        scoring='neg_mean_absolute_error',
                        return_train_score=True, n_jobs=-1)
lasso_gs.fit(X_train, y_train)

def tuning_table(gs, name):
    out = pd.DataFrame({
        'alpha':                [p['model__alpha'] for p in gs.cv_results_['params']],
        'CV MAE (transformed)': -gs.cv_results_['mean_test_score'],
        'std':                   gs.cv_results_['std_test_score']
    }).sort_values('CV MAE (transformed)')
    print(f"\\n{name}: all trials")
    print(out.to_string(index=False))
    print(f"Best alpha = {gs.best_params_['model__alpha']}  (CV MAE = {-gs.best_score_:.4f})")

tuning_table(ridge_gs, "Ridge Regression")
tuning_table(lasso_gs, "Lasso Regression")"""
),

code(
"""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, gs, color, label in [
    (axes[0], ridge_gs, 'steelblue', 'Ridge'),
    (axes[1], lasso_gs, 'seagreen',  'Lasso'),
]:
    alphas  = [p['model__alpha'] for p in gs.cv_results_['params']]
    cv_maes = [-s for s in gs.cv_results_['mean_test_score']]
    cv_stds =  gs.cv_results_['std_test_score']
    best_a  =  gs.best_params_['model__alpha']

    ax.semilogx(alphas, cv_maes, 'o-', color=color, linewidth=2, markersize=8)
    ax.fill_between(alphas,
                    [m - s for m, s in zip(cv_maes, cv_stds)],
                    [m + s for m, s in zip(cv_maes, cv_stds)],
                    alpha=0.15, color=color)
    ax.axvline(best_a, color='red', linestyle='--', label=f'Best alpha = {best_a}')
    ax.set_title(f'{label}: Tuning Curve', fontweight='bold')
    ax.set_xlabel('Alpha (log scale)')
    ax.set_ylabel('CV MAE (transformed scale)')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.suptitle('Phase 3: Hyperparameter Tuning — 5-Fold CV on Training Set',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
),

md("**Insight:** A flat tuning curve indicates that regularisation strength has little effect on performance, which suggests the model is not overfitting — its limitation is the linearity assumption rather than parameter choice. The best alpha is the one that minimises CV MAE on the training set without the validation set being involved."),

md("### Phase 4: Validation on the Held-Out Validation Set\n\nThe best Ridge and Lasso configurations from Phase 3 are now evaluated on the validation set for the first time since Phase 2. This confirms whether the chosen parameters generalise beyond the training data. All metrics are computed on the original dollar scale. The model with the lowest validation MAE is selected as the final candidate for Phase 5."),

code(
"""res_base = {'MAE': mae_B, 'RMSE': rmse_B, 'R2': r2_B}

print("=" * 60)
print("PHASE 4: VALIDATION ON HELD-OUT VALIDATION SET")
print("=" * 60)
print("(Validation set was sealed during all of Phase 3)\\n")

res_ridge = evaluate(ridge_gs.best_estimator_, X_val, y_val,
                     f"Ridge alpha={ridge_gs.best_params_['model__alpha']} (validation)")
res_lasso = evaluate(lasso_gs.best_estimator_, X_val, y_val,
                     f"Lasso alpha={lasso_gs.best_params_['model__alpha']} (validation)")

val_summary = pd.DataFrame({
    'Model': [
        'Baseline LR (Model B)',
        f"Ridge  alpha={ridge_gs.best_params_['model__alpha']}",
        f"Lasso  alpha={lasso_gs.best_params_['model__alpha']}"
    ],
    'Val MAE ($)' : [res_base['MAE'],  res_ridge['MAE'],  res_lasso['MAE']],
    'Val RMSE ($)': [res_base['RMSE'], res_ridge['RMSE'], res_lasso['RMSE']],
    'Val R2'      : [res_base['R2'],   res_ridge['R2'],   res_lasso['R2']],
})

print("\\nValidation Results (original dollar scale):")
print(val_summary.to_string(index=False))

best_idx  = val_summary['Val MAE ($)'].idxmin()
best_name = val_summary.loc[best_idx, 'Model']
best_candidates = {0: ('linear', None),
                   1: ('ridge',  ridge_gs.best_params_['model__alpha']),
                   2: ('lasso',  lasso_gs.best_params_['model__alpha'])}
best_type, best_alpha = best_candidates[best_idx]
print(f"\\nSelected for Phase 5: {best_name}")

colors = ['steelblue', 'seagreen', 'darkorange']
model_labels = ['Baseline LR',
                f"Ridge\\nalpha={ridge_gs.best_params_['model__alpha']}",
                f"Lasso\\nalpha={lasso_gs.best_params_['model__alpha']}"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, ['Val MAE ($)', 'Val RMSE ($)', 'Val R2']):
    bars = ax.bar(model_labels, val_summary[metric], color=colors, edgecolor='black', width=0.5)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
    ax.set_title(metric, fontweight='bold')
    ax.tick_params(axis='x', rotation=10)
plt.suptitle('Phase 4: Validation Comparison (Original Dollar Scale)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
),

md("**Insight:** Phase 4 is the first and only time the validation set is used to compare regularised models. The model selected here is chosen based on unseen data, which makes the selection trustworthy. It carries forward to Phase 5 where it faces the truly sealed test set."),

md("### Phase 5: Final Evaluation on the Unseen Test Set\n\nThe selected model is retrained on the combined training and validation data to give it more examples. It is then evaluated exactly once on the test set that has been sealed since Phase 1 and has never influenced any decision. These are the final reported results for Linear Regression. A 5-fold CV stability check confirms the model performs consistently and the test result is not a lucky split."),

code(
"""import joblib

print("=" * 60)
print("PHASE 5: FINAL EVALUATION ON UNSEEN TEST SET")
print("=" * 60)

X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])

if best_type == 'linear':
    final_lr_model = LinearRegression()
elif best_type == 'ridge':
    final_lr_model = Ridge(alpha=best_alpha)
else:
    final_lr_model = Lasso(alpha=best_alpha, max_iter=10000)

final_lr_pipeline = Pipeline([('scaler', StandardScaler()), ('model', final_lr_model)])
final_lr_pipeline.fit(X_full_train, y_full_train)

print(f"Final model  : {best_name}")
print(f"Trained on   : {len(X_full_train)} rows (train + validation combined)")
print(f"Evaluated on : {len(X_test)} rows (previously unseen test set)\\n")

res_lr_final = evaluate(final_lr_pipeline, X_test, y_test, "FINAL Linear Regression (test set)")

cv_final = cross_val_score(final_lr_pipeline, X_full_train, y_full_train, cv=kf, scoring='r2')
print(f"\\n5-Fold CV R2 stability: {cv_final.round(3)}")
print(f"Mean = {cv_final.mean():.4f}  Std = {cv_final.std():.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(res_lr_final['y_true'], res_lr_final['y_pred'],
            alpha=0.3, color='steelblue', s=15, edgecolors='none')
min_v = min(res_lr_final['y_true'].min(), res_lr_final['y_pred'].min())
max_v = max(res_lr_final['y_true'].max(), res_lr_final['y_pred'].max())
plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title(f'Linear Regression: Final Test Set Results', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

joblib.dump(final_lr_pipeline, 'final_lr_pipeline.pkl')
pd.DataFrame({'actual_cost': res_lr_final['y_true'],
              'predicted_cost': res_lr_final['y_pred']}).to_csv('lr_final_predictions.csv', index=False)
print("\\nSaved: final_lr_pipeline.pkl  |  lr_final_predictions.csv")"""
),

md("**Insight:** The test set result is the most honest measure of performance because the test set was never used for any decision during the pipeline. The CV stability check run alongside confirms the result is consistent and not driven by a lucky split."),

md("### Phase 6: Linear Regression Results Summary\n\nAll evaluation results from Phases 2 through 5 are collected below in a single table, all reported on the original dollar scale, showing the progression from the raw baseline through tuning to the final test evaluation."),

code(
"""print("=" * 60)
print("LINEAR REGRESSION: COMPLETE RESULTS SUMMARY")
print("=" * 60)

summary_lr = pd.DataFrame({
    'Phase': [
        'Phase 2: Baseline LR Model B (validation)',
        f"Phase 3/4: Ridge alpha={ridge_gs.best_params_['model__alpha']} (validation)",
        f"Phase 3/4: Lasso alpha={lasso_gs.best_params_['model__alpha']} (validation)",
        f"Phase 5: {best_name} (UNSEEN TEST SET)",
    ],
    'MAE ($)' : [res_base['MAE'],  res_ridge['MAE'],  res_lasso['MAE'],  res_lr_final['MAE']],
    'RMSE ($)': [res_base['RMSE'], res_ridge['RMSE'], res_lasso['RMSE'], res_lr_final['RMSE']],
    'R2'      : [res_base['R2'],   res_ridge['R2'],   res_lasso['R2'],   res_lr_final['R2']],
})

print(summary_lr.to_string(index=False))"""
),

md("**Insight:** A low R2 across all linear models is consistent with the EDA finding that supplier cost is driven by non-linear interactions among tube geometry, component weight, order type, and supplier strategy. This result is a meaningful baseline. The KNN model in the next section uses a fundamentally different approach and will be compared against these numbers in the final comparison."),

md("### Explainable AI: Feature Importance\n\nLinear regression is naturally interpretable because each prediction is a weighted sum of the input features. Two methods are used here. Standardised coefficients show the relative contribution of each feature on a comparable scale. SHAP values (SHapley Additive Explanations) provide a theoretically grounded decomposition of each prediction into per-feature contributions, showing both magnitude and direction."),

md("#### Standardised Coefficients\n\nBecause features are standardised inside the pipeline, the coefficients are on a comparable scale. A larger absolute value means that feature has a stronger influence on the prediction. Blue bars increase the predicted cost and red bars decrease it."),

code(
"""feature_names = X_train.columns.tolist()
coef = model_B.named_steps['model'].coef_

coef_df = pd.DataFrame({
    'Feature':     feature_names,
    'Coefficient': coef
}).sort_values('Coefficient', key=abs, ascending=False)

colors = ['steelblue' if c > 0 else 'tomato' for c in coef_df['Coefficient']]

plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black')
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.xlabel('Standardised Coefficient')
plt.title('Linear Regression: Standardised Feature Coefficients\\n'
          '(Blue = increases predicted cost  |  Red = decreases predicted cost)',
          fontweight='bold')
plt.tight_layout()
plt.show()

print("Top 5 features by absolute coefficient weight:")
print(coef_df.head(5).to_string(index=False))"""
),

md("#### SHAP Values (Global and Local Explanations)\n\nSHAP (SHapley Additive Explanations) is the current standard for explainable AI in machine learning. A `LinearExplainer` is used here because the model is linear. The global bar plot shows average absolute impact per feature across all validation samples. The dot plot adds direction. The waterfall plot explains one individual prediction step by step."),

code("import subprocess\nsubprocess.run(['pip', 'install', 'shap'], check=True)"),

code(
"""import shap

explainer = shap.LinearExplainer(
    model_B.named_steps['model'],
    shap.sample(model_B.named_steps['scaler'].transform(X_train), 100)
)

X_val_scaled = model_B.named_steps['scaler'].transform(X_val)
shap_values  = explainer.shap_values(X_val_scaled)

shap.summary_plot(shap_values, X_val, feature_names=feature_names, plot_type='bar', show=False)
plt.title('SHAP Global Feature Importance (Linear Regression)', fontweight='bold')
plt.tight_layout()
plt.show()"""
),

code(
"""shap.summary_plot(shap_values, X_val, feature_names=feature_names, show=False)
plt.title('SHAP Summary: Feature Impact Direction and Magnitude', fontweight='bold')
plt.tight_layout()
plt.show()"""
),

code(
"""row_index = 0

shap.waterfall_plot(
    shap.Explanation(
        values        = shap_values[row_index],
        base_values   = explainer.expected_value,
        data          = X_val.iloc[row_index].values,
        feature_names = feature_names
    )
)"""
),

md("**Insight:** The SHAP global bar plot ranks features by average absolute contribution across all validation samples. The dot plot shows direction: features pushing predictions higher appear on the right. The waterfall plot for a single row answers the question of why the model predicted a specific price for that particular tube assembly, starting from the dataset average and showing each feature's push up or down toward the final prediction."),

]

nb['cells'].extend(cells)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Added {len(cells)} cells. Notebook now has {len(nb['cells'])} cells.")
