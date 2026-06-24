import json, uuid

NB = r"c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def md(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}
def code(src):
    return {"cell_type":"code","execution_count":None,"id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}

cells = [

md("## 3. KNN Regression\n\nThis section applies K-Nearest Neighbours Regression (KNN) as the second modelling algorithm. KNN is a non-parametric method that makes no assumption about the shape of the relationship between features and the target. Instead of fitting a global equation, it predicts the cost of a new tube assembly by looking at the K most similar assemblies in the training data and averaging their costs. This makes KNN fundamentally different from Linear Regression and a useful comparison. The same data split, evaluation function, and metric scale from the Linear Regression section are reused throughout."),

md("### Phase 2: KNN Baseline\n\nA baseline KNN model is trained with the default K=5 on the power-transformed cost. `StandardScaler` is applied inside the pipeline because KNN is a distance-based algorithm — without scaling, features measured in large units (such as `annual_usage`) would dominate the distance calculation and overpower smaller features such as `wall` thickness. Predictions are back-transformed to dollars for evaluation using the same `evaluate` function defined earlier."),

code(
"""knn_base = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  KNeighborsRegressor(n_neighbors=5))
])
knn_base.fit(X_train, y_train)

print("=== KNN Baseline (K=5, Power-Transformed Cost) ===")
res_knn_base = evaluate(knn_base, X_val, y_val, "KNN baseline K=5 (validation)")"""
),

md("#### KNN Baseline: 5-Fold Cross-Validation\n\nThe same KFold setup from the Linear Regression section is applied here. Cross-validation on the training set with K=5 gives an initial view of KNN's stability before any tuning."),

code(
"""cv_knn = cross_validate(
    knn_base, X_train, y_train,
    cv=kf,
    scoring={'r2': 'r2', 'mae': 'neg_mean_absolute_error',
             'rmse': 'neg_root_mean_squared_error'}
)

cv_r2_knn   =  cv_knn['test_r2'].mean()
cv_mae_knn  = -cv_knn['test_mae'].mean()
cv_rmse_knn = -cv_knn['test_rmse'].mean()

print("=== KNN Baseline: 5-Fold Cross-Validation (K=5) ===")
print(f"CV MAE  : {cv_mae_knn:.4f}  (transformed scale)")
print(f"CV RMSE : {cv_rmse_knn:.4f}  (transformed scale)")
print(f"CV R2   : {cv_r2_knn:.4f}")
print(f"All CV R2 scores: {cv_knn['test_r2'].round(3)}")"""
),

code(
"""y_val_orig = (y_val ** 20).reset_index(drop=True)
y_pred_knn_base = np.clip(knn_base.predict(X_val), 0, None) ** 20

plt.figure(figsize=(8, 5))
plt.scatter(y_val_orig, y_pred_knn_base, alpha=0.3, color='mediumpurple',
            edgecolors='none', s=15)
plt.plot([y_val_orig.min(), y_val_orig.max()],
         [y_val_orig.min(), y_val_orig.max()],
         'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title('KNN Baseline (K=5): Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()"""
),

md("**Insight:** The baseline KNN with K=5 gives an initial measure of how well a nearest-neighbour approach handles this dataset. Small K values can overfit to local noise in the training data, which is why Phase 3 searches for the optimal K across a wider range. The scatter plot shows whether KNN captures the general trend of increasing cost with larger and heavier assemblies."),

md("### Phase 3: Hyperparameter Tuning — Finding the Best K\n\nThe only hyperparameter for KNN is `n_neighbors` (K), the number of nearest neighbours used to compute each prediction. A small K makes the model sensitive to individual training points (high variance). A large K smooths over local patterns and risks underfitting (high bias). GridSearchCV with 5-fold cross-validation is used to find the K that minimises CV MAE on the training set. The validation set remains sealed during this entire phase."),

code(
"""print("=" * 60)
print("PHASE 3: KNN HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

knn_pipe   = Pipeline([('scaler', StandardScaler()), ('model', KNeighborsRegressor())])
knn_params = {'model__n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30]}

knn_gs = GridSearchCV(
    knn_pipe, knn_params,
    cv=kf, scoring='neg_mean_absolute_error',
    return_train_score=True, n_jobs=-1
)
knn_gs.fit(X_train, y_train)

knn_results = pd.DataFrame({
    'K':                    [p['model__n_neighbors'] for p in knn_gs.cv_results_['params']],
    'CV MAE (transformed)': -knn_gs.cv_results_['mean_test_score'],
    'std':                   knn_gs.cv_results_['std_test_score']
}).sort_values('CV MAE (transformed)')

print("\\nKNN: all trials")
print(knn_results.to_string(index=False))
print(f"\\nBest K = {knn_gs.best_params_['model__n_neighbors']}  "
      f"(CV MAE = {-knn_gs.best_score_:.4f})")"""
),

code(
"""k_values = [p['model__n_neighbors'] for p in knn_gs.cv_results_['params']]
cv_maes  = [-s for s in knn_gs.cv_results_['mean_test_score']]
cv_stds  =  knn_gs.cv_results_['std_test_score']
best_k   =  knn_gs.best_params_['model__n_neighbors']

plt.figure(figsize=(10, 5))
plt.plot(k_values, cv_maes, 'o-', color='mediumpurple', linewidth=2, markersize=8)
plt.fill_between(k_values,
                 [m - s for m, s in zip(cv_maes, cv_stds)],
                 [m + s for m, s in zip(cv_maes, cv_stds)],
                 alpha=0.15, color='mediumpurple')
plt.axvline(best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
plt.xlabel('Number of Neighbours (K)')
plt.ylabel('CV MAE (transformed scale)')
plt.title('KNN: Tuning Curve — CV MAE vs K', fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""
),

md("**Insight:** The tuning curve shows how CV MAE changes as K increases. Very small K values typically produce higher error due to overfitting to local noise. As K increases, the model smooths out and error decreases until it reaches a minimum, after which adding more neighbours starts to underfit. The best K is the value at the lowest point of the curve."),

md("### Phase 4: Validation of the Best K on the Held-Out Validation Set\n\nThe best K found in Phase 3 is now applied to the validation set for the first time. This confirms whether the chosen K generalises beyond the training data. All metrics are on the original dollar scale for a direct comparison with the Linear Regression results from Phase 4 of the previous section."),

code(
"""print("=" * 60)
print("PHASE 4: KNN VALIDATION ON HELD-OUT VALIDATION SET")
print("=" * 60)
print("(Validation set was sealed during all of Phase 3)\\n")

res_knn_val = evaluate(knn_gs.best_estimator_, X_val, y_val,
                       f"KNN best K={best_k} (validation)")

print(f"\\nKNN best K={best_k} vs KNN baseline K=5:")
print(f"  Baseline K=5  MAE=${res_knn_base['MAE']:.4f}  RMSE=${res_knn_base['RMSE']:.4f}  R2={res_knn_base['R2']:.4f}")
print(f"  Best    K={best_k:<3}  MAE=${res_knn_val['MAE']:.4f}  RMSE=${res_knn_val['RMSE']:.4f}  R2={res_knn_val['R2']:.4f}")

y_pred_knn_val = np.clip(knn_gs.best_estimator_.predict(X_val), 0, None) ** 20
y_val_orig     = (y_val ** 20).reset_index(drop=True)

plt.figure(figsize=(8, 5))
plt.scatter(y_val_orig, y_pred_knn_val, alpha=0.3, color='mediumpurple',
            edgecolors='none', s=15)
plt.plot([y_val_orig.min(), y_val_orig.max()],
         [y_val_orig.min(), y_val_orig.max()],
         'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title(f'KNN (Best K={best_k}): Actual vs Predicted (Validation Set)')
plt.legend()
plt.tight_layout()
plt.show()"""
),

md("**Insight:** Comparing the best tuned K against the baseline K=5 on the validation set shows whether tuning made a meaningful difference. The validation result here is also the first opportunity to compare KNN directly against the Linear Regression models from Phase 4 of the previous section, though the formal side-by-side comparison is reserved for Section 4."),

md("### Phase 5: Final Evaluation on the Unseen Test Set\n\nThe best KNN model is retrained on the combined training and validation data. It is then evaluated exactly once on the sealed test set. These are the final reported results for KNN Regression. A CV stability check confirms the model is consistent and the test result is not a lucky split."),

code(
"""print("=" * 60)
print("PHASE 5: KNN FINAL EVALUATION ON UNSEEN TEST SET")
print("=" * 60)

final_knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  KNeighborsRegressor(n_neighbors=best_k))
])
final_knn_pipeline.fit(X_full_train, y_full_train)

print(f"Final KNN model  : K = {best_k}")
print(f"Trained on       : {len(X_full_train)} rows (train + validation combined)")
print(f"Evaluated on     : {len(X_test)} rows (previously unseen test set)\\n")

res_knn_final = evaluate(final_knn_pipeline, X_test, y_test,
                         f"FINAL KNN K={best_k} (test set)")

cv_knn_final = cross_val_score(final_knn_pipeline, X_full_train, y_full_train,
                                cv=kf, scoring='r2')
print(f"\\n5-Fold CV R2 stability: {cv_knn_final.round(3)}")
print(f"Mean = {cv_knn_final.mean():.4f}  Std = {cv_knn_final.std():.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(res_knn_final['y_true'], res_knn_final['y_pred'],
            alpha=0.3, color='mediumpurple', s=15, edgecolors='none')
min_v = min(res_knn_final['y_true'].min(), res_knn_final['y_pred'].min())
max_v = max(res_knn_final['y_true'].max(), res_knn_final['y_pred'].max())
plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect prediction')
plt.xlabel('Actual Cost ($)')
plt.ylabel('Predicted Cost ($)')
plt.title(f'KNN (K={best_k}): Final Test Set Results', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

joblib.dump(final_knn_pipeline, 'final_knn_pipeline.pkl')
pd.DataFrame({'actual_cost': res_knn_final['y_true'],
              'predicted_cost': res_knn_final['y_pred']}).to_csv('knn_final_predictions.csv', index=False)
print("\\nSaved: final_knn_pipeline.pkl  |  knn_final_predictions.csv")"""
),

md("**Insight:** The test set result for KNN, like that of Linear Regression, is evaluated exactly once after the pipeline is complete. Comparing the CV mean and standard deviation with the test score shows whether the model is stable or whether it performs differently on the final held-out data."),

md("### Phase 6: KNN Results Summary"),

code(
"""print("=" * 60)
print("KNN REGRESSION: COMPLETE RESULTS SUMMARY")
print("=" * 60)

summary_knn = pd.DataFrame({
    'Phase': [
        'Phase 2: KNN Baseline K=5 (validation)',
        f"Phase 3/4: KNN Best K={best_k} (validation)",
        f"Phase 5: KNN K={best_k} (UNSEEN TEST SET)",
    ],
    'MAE ($)' : [res_knn_base['MAE'], res_knn_val['MAE'], res_knn_final['MAE']],
    'RMSE ($)': [res_knn_base['RMSE'], res_knn_val['RMSE'], res_knn_final['RMSE']],
    'R2'      : [res_knn_base['R2'], res_knn_val['R2'], res_knn_final['R2']],
})

print(summary_knn.to_string(index=False))"""
),

md("**Insight:** The KNN summary shows how performance changed from the untuned baseline to the best K found by GridSearchCV, and finally to the result on the unseen test set. A reduction in MAE from baseline to best K confirms that tuning made a meaningful improvement."),

md("## 4. Final Model Comparison: Linear Regression vs KNN\n\nBoth models have now been trained, tuned, and evaluated on the same unseen test set using the same feature set, transformation, and evaluation function. This section brings the final results together to identify the stronger model and explain what the comparison reveals about the nature of the supplier pricing prediction problem."),

code(
"""print("=" * 60)
print("FINAL MODEL COMPARISON: LINEAR REGRESSION vs KNN")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': [
        f"Linear Regression ({best_name})",
        f"KNN Regression (K={best_k})"
    ],
    'Test MAE ($)' : [res_lr_final['MAE'],  res_knn_final['MAE']],
    'Test RMSE ($)': [res_lr_final['RMSE'], res_knn_final['RMSE']],
    'Test R2'      : [res_lr_final['R2'],   res_knn_final['R2']],
})

print("\\nFinal Test Set Results (original dollar scale):")
print(comparison.to_string(index=False))

winner = comparison.loc[comparison['Test MAE ($)'].idxmin(), 'Model']
print(f"\\nBest model by Test MAE: {winner}")

metrics = ['Test MAE ($)', 'Test RMSE ($)', 'Test R2']
colors  = ['steelblue', 'mediumpurple']
labels  = [f"Linear Regression\\n({best_name})", f"KNN\\n(K={best_k})"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for ax, metric in zip(axes, metrics):
    bars = ax.bar(labels, comparison[metric], color=colors, edgecolor='black', width=0.4)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=10)
    ax.set_title(metric, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
plt.suptitle('Final Comparison: Linear Regression vs KNN (Test Set, Original Dollar Scale)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()"""
),

md("**Insight:** The comparison reveals which approach — a global linear model or a local distance-based model — better captures the pricing patterns in the Caterpillar tube assembly dataset. Linear Regression assumes a straight-line relationship between features and cost, which may miss non-linear patterns. KNN makes no such assumption and can capture local pricing clusters, but it is sensitive to the choice of K and to noisy features. The model with the lower test MAE is the recommended choice for deployment, but both results together demonstrate that supplier pricing prediction benefits from further exploration with non-linear methods such as gradient boosting or random forests."),

]

nb['cells'].extend(cells)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Added {len(cells)} cells. Notebook now has {len(nb['cells'])} cells.")
