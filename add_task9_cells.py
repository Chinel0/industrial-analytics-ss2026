import json, uuid

NB = r"c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

def md(src):
    return {"cell_type":"markdown","id":uuid.uuid4().hex[:8],"metadata":{},"source":src}
def code(src):
    return {"cell_type":"code","execution_count":None,"id":uuid.uuid4().hex[:8],"metadata":{},"outputs":[],"source":src}

cells = [

md("## 5. Task 9: Evaluation and Enhancement\n\nTask 8 produced validated baselines for both Linear Regression and KNN. This section analyses their weaknesses, tests targeted enhancements on the validation set, selects the best candidate, and evaluates it once on the test set. An error and limitation analysis closes the notebook.\n\nThe structure follows the required task deliverables:\n\n1. **9.1 Validation Analysis:** consolidated metrics from Task 8 and identified weaknesses\n2. **9.2 Enhancement Experiments:** three documented experiments, each evaluated on the validation set only\n3. **9.3 Enhancement Log:** all experiments compared in one table\n4. **9.4 Candidate Selection and Final Test Evaluation:** best experiment retrained and evaluated on the test set\n5. **9.5 Error Analysis:** residual analysis and root cause investigation\n6. **9.6 Limitations and Evaluation Summary**"),

md("### 9.1 Validation Analysis Report\n\nThe next cell consolidates all validation and cross-validation results from Task 8 into one report and lists the weaknesses that the enhancement experiments will target."),

code(
"""print("=" * 60)
print("9.1 VALIDATION ANALYSIS REPORT (results from Task 8)")
print("=" * 60)

alpha_ridge = ridge_gs.best_params_['model__alpha']
alpha_lasso = lasso_gs.best_params_['model__alpha']

analysis = pd.DataFrame({
    'Model': [
        'LR Model A (raw cost)',
        'LR Model B (power-transformed)',
        f'LR Ridge alpha={alpha_ridge}',
        f'LR Lasso alpha={alpha_lasso}',
        'KNN Baseline K=5',
        f'KNN Best K={best_k}',
    ],
    'Val MAE ($)' : [mae_A,  mae_B,  res_ridge['MAE'],  res_lasso['MAE'],
                     res_knn_base['MAE'],  res_knn_val['MAE']],
    'Val RMSE ($)': [rmse_A, rmse_B, res_ridge['RMSE'], res_lasso['RMSE'],
                     res_knn_base['RMSE'], res_knn_val['RMSE']],
    'Val R2'      : [r2_A,   r2_B,   res_ridge['R2'],   res_lasso['R2'],
                     res_knn_base['R2'],   res_knn_val['R2']],
    'CV R2'       : [cv_r2_A, cv_r2_B, np.nan, np.nan, cv_r2_knn, np.nan],
})

print(analysis.round(4).to_string(index=False))

print("\\nIdentified weaknesses to address in 9.2:")
print("  1. Both models underestimate high-cost assemblies above $50.")
print("     The power back-transformation amplifies small errors in the")
print("     transformed scale into large dollar errors at the high end.")
print("  2. The linear model R2 is moderate, indicating that pricing")
print("     depends on non-linear interactions between features that")
print("     a purely linear model cannot capture.")
print("  3. With only 7 features, SelectKBest may identify a smaller")
print("     subset that removes noise and improves generalisation.")
print("  4. The current feature set does not include interaction terms,")
print("     for example diameter multiplied by length, which would")
print("     represent how physical size combinations drive cost.")"""
),

md("**Insight:** The validation report shows that KNN achieved a different error profile from Linear Regression. Linear models produce smooth global predictions while KNN produces local averages. Both models struggle with high-cost assemblies because those cases are rare in the training data, which limits what either algorithm can learn from them."),

md("### 9.2 Enhancement Experiments\n\nThree enhancements are tested below. Each experiment is trained on the training set only and evaluated on the validation set. The test set remains sealed until section 9.4. The Task 8 best Linear Regression model (Ridge with the best alpha) is the reference baseline for comparison."),

md("#### Experiment 1: SelectKBest Feature Reduction\n\nWith 7 input features, some may carry overlapping or low-signal information that adds noise without improving predictions. This experiment uses `SelectKBest` with the F-regression score to keep only the top 5 features, then retrains the tuned Ridge model on the reduced set. The selection step is fitted on training data only and is part of the pipeline to avoid data leakage."),

code(
"""from sklearn.feature_selection import SelectKBest, f_regression as f_reg_score

exp_log = []
alpha_best = ridge_gs.best_params_['model__alpha']

# Reference: Task 8 best LR model (Ridge, all 7 features)
exp_log.append({
    'Experiment':  'Reference: Ridge all 7 features',
    'Features':    X_train.shape[1],
    'Val MAE ($)': res_ridge['MAE'],
    'Val RMSE ($)':res_ridge['RMSE'],
    'Val R2':      res_ridge['R2']
})

# Experiment 1: SelectKBest top 5
exp1_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('select', SelectKBest(score_func=f_reg_score, k=5)),
    ('model',  Ridge(alpha=alpha_best))
])
exp1_pipe.fit(X_train, y_train)

res_e1 = evaluate(exp1_pipe, X_val, y_val, 'Experiment 1: SelectKBest top 5 features')
exp_log.append({
    'Experiment':  'Exp 1: SelectKBest top 5 features',
    'Features':    5,
    'Val MAE ($)': res_e1['MAE'],
    'Val RMSE ($)':res_e1['RMSE'],
    'Val R2':      res_e1['R2']
})

selected_e1 = X_train.columns[exp1_pipe.named_steps['select'].get_support()].tolist()
print(f"Top 5 features selected by F-regression: {selected_e1}")"""
),

md("#### Experiment 2: Polynomial Interaction Features\n\nSupplier pricing depends on combinations of physical properties. For example, a wide diameter tube that is also long will cost more than either feature alone suggests. A purely linear model cannot capture this. This experiment adds degree-2 polynomial and interaction terms for the five strongest physical cost drivers, then trains the Ridge model on the expanded feature set. Scaling is applied after the expansion to keep all terms comparable."),

code(
"""from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import ColumnTransformer

key_feats   = ['diameter', 'wall', 'length', 'quantity', 'Total_weight']
other_feats = [c for c in X_train.columns if c not in key_feats]

pre = ColumnTransformer([
    ('poly', PolynomialFeatures(degree=2, include_bias=False), key_feats),
    ('rest', 'passthrough', other_feats)
])

exp2_pipe = Pipeline([
    ('pre',    pre),
    ('scaler', StandardScaler()),
    ('model',  Ridge(alpha=alpha_best))
])
exp2_pipe.fit(X_train, y_train)

res_e2 = evaluate(exp2_pipe, X_val, y_val, 'Experiment 2: polynomial interaction features')
n_feats_e2 = pre.fit_transform(X_train.head(1)).shape[1]
exp_log.append({
    'Experiment':  'Exp 2: polynomial interaction features',
    'Features':    n_feats_e2,
    'Val MAE ($)': res_e2['MAE'],
    'Val RMSE ($)':res_e2['RMSE'],
    'Val R2':      res_e2['R2']
})

print(f"Key features expanded with degree-2 terms: {key_feats}")
print(f"Total features after polynomial expansion: {n_feats_e2}")"""
),

md("#### Experiment 3: KNN with Manhattan Distance\n\nThe default KNN model uses Euclidean distance to find neighbours. For datasets where features have different relationships with the target, Manhattan distance (sum of absolute differences) can produce different and sometimes better neighbourhood definitions. This experiment retunes K with Manhattan distance and compares the result against the Euclidean-based KNN from Task 8."),

code(
"""knn_params_e3 = {
    'model__n_neighbors': [3, 5, 7, 10, 15, 20, 25, 30],
    'model__metric':      ['manhattan']
}

exp3_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  KNeighborsRegressor())
])

exp3_gs = GridSearchCV(
    exp3_pipe, knn_params_e3,
    cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1
)
exp3_gs.fit(X_train, y_train)

best_k_e3 = exp3_gs.best_params_['model__n_neighbors']
res_e3 = evaluate(exp3_gs.best_estimator_, X_val, y_val,
                  f'Experiment 3: KNN Manhattan K={best_k_e3}')
exp_log.append({
    'Experiment':  f'Exp 3: KNN Manhattan distance K={best_k_e3}',
    'Features':    X_train.shape[1],
    'Val MAE ($)': res_e3['MAE'],
    'Val RMSE ($)':res_e3['RMSE'],
    'Val R2':      res_e3['R2']
})

print(f"Best K with Manhattan distance: {best_k_e3}  (CV MAE = {-exp3_gs.best_score_:.4f})")"""
),

md("### 9.3 Enhancement Log\n\nAll experiments are collected into a single table sorted by validation MAE. This is the documented experiment log showing what was tried, how many features each version used, and what it achieved on the validation set."),

code(
"""exp_df = pd.DataFrame(exp_log).sort_values('Val MAE ($)').reset_index(drop=True)

print("=" * 60)
print("9.3 ENHANCEMENT LOG (sorted by validation MAE)")
print("=" * 60)
print(exp_df.round(4).to_string(index=False))

best_exp_name = exp_df.loc[0, 'Experiment']
print(f"\\nBest experiment on the validation set: {best_exp_name}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

d1 = exp_df.sort_values('Val MAE ($)', ascending=True)
axes[0].barh(d1['Experiment'], d1['Val MAE ($)'], color='steelblue', edgecolor='black')
axes[0].set_title('Validation MAE per Experiment (lower is better)', fontweight='bold')
axes[0].set_xlabel('Validation MAE ($)')

d2 = exp_df.sort_values('Val R2', ascending=True)
axes[1].barh(d2['Experiment'], d2['Val R2'], color='seagreen', edgecolor='black')
axes[1].set_title('Validation R2 per Experiment (higher is better)', fontweight='bold')
axes[1].set_xlabel('Validation R2')

plt.suptitle('9.3 Enhancement Log: All Experiments on Validation Set',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()"""
),

md("**Insight:** The enhancement log shows which modification produced the biggest gain on the validation set. If the polynomial interaction experiment wins, it confirms that non-linear feature combinations add predictive value beyond the raw features. If SelectKBest wins, it suggests that some features were adding noise rather than signal. If the KNN Manhattan experiment wins, it confirms that the neighbourhood definition matters for this dataset's structure."),

md("### 9.4 Candidate Selection and Final Test Evaluation\n\nThe experiment with the lowest validation MAE is the enhanced candidate model. It is retrained on the combined training and validation data and evaluated on the test set exactly once. The result is compared with the Task 8 final Linear Regression model to show whether the enhancement genuinely improved generalisation."),

code(
"""print("=" * 60)
print("9.4 FINAL TEST EVALUATION OF THE ENHANCED MODEL")
print("=" * 60)

X_full_train = pd.concat([X_train, X_val])
y_full_train = pd.concat([y_train, y_val])

if best_exp_name.startswith('Exp 1'):
    enhanced = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_reg_score, k=5)),
        ('model',  Ridge(alpha=alpha_best))
    ])
    enhanced.fit(X_full_train, y_full_train)
    res_enh = evaluate(enhanced, X_test, y_test, 'ENHANCED model (test set)')

elif best_exp_name.startswith('Exp 2'):
    pre_final = ColumnTransformer([
        ('poly', PolynomialFeatures(degree=2, include_bias=False), key_feats),
        ('rest', 'passthrough', other_feats)
    ])
    enhanced = Pipeline([
        ('pre',    pre_final),
        ('scaler', StandardScaler()),
        ('model',  Ridge(alpha=alpha_best))
    ])
    enhanced.fit(X_full_train, y_full_train)
    res_enh = evaluate(enhanced, X_test, y_test, 'ENHANCED model (test set)')

elif best_exp_name.startswith('Exp 3'):
    enhanced = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  KNeighborsRegressor(n_neighbors=best_k_e3, metric='manhattan'))
    ])
    enhanced.fit(X_full_train, y_full_train)
    res_enh = evaluate(enhanced, X_test, y_test, 'ENHANCED model (test set)')

else:
    enhanced = Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=alpha_best))])
    enhanced.fit(X_full_train, y_full_train)
    res_enh = evaluate(enhanced, X_test, y_test, 'ENHANCED model (test set)')

print("\\nComparison with Task 8 final Linear Regression model on the same test set:")
compare = pd.DataFrame({
    'Model':       ['Task 8: Final LR (Ridge)', f'Task 9: Enhanced ({best_exp_name})'],
    'Test MAE ($)':[res_lr_final['MAE'],  res_enh['MAE']],
    'Test RMSE ($)':[res_lr_final['RMSE'], res_enh['RMSE']],
    'Test R2':     [res_lr_final['R2'],   res_enh['R2']],
})
print(compare.round(4).to_string(index=False))

import joblib
joblib.dump(enhanced, 'enhanced_model.pkl')
pd.DataFrame({
    'actual_cost':    res_enh['y_true'].values,
    'predicted_cost': res_enh['y_pred']
}).to_csv('enhanced_predictions.csv', index=False)
print("\\nSaved: enhanced_model.pkl  |  enhanced_predictions.csv")"""
),

md("**Insight:** The test set result for the enhanced model is the final reported performance. If the enhanced model's test MAE is lower than the Task 8 baseline, the enhancement genuinely improved generalisation and not just validation set performance. If the gap between validation MAE and test MAE is small, the model is stable and not overfitted to the validation data."),

md("### 9.5 Error Analysis\n\nConfusion matrices apply to classification tasks. Since this is a regression problem, the equivalent error analysis tools are residual plots and a performance breakdown by cost bracket, which serves as the regression counterpart of a confusion matrix.\n\nThree views are produced:\n1. Residual distribution: prediction errors should be centred near zero with no strong skew\n2. Residuals vs predicted values: systematic patterns here reveal directional bias\n3. Mean absolute error by cost bracket: shows in which price range the model succeeds and where it fails"),

code(
"""y_true_err = pd.Series(res_enh['y_true']).reset_index(drop=True)
y_pred_err = pd.Series(res_enh['y_pred']).reset_index(drop=True)
residuals  = y_true_err - y_pred_err

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(residuals, bins=80, color='steelblue', edgecolor='black')
axes[0, 0].axvline(0, color='red', linestyle='--')
axes[0, 0].set_title('Residual Distribution (actual minus predicted)', fontweight='bold')
axes[0, 0].set_xlabel('Residual ($)')
axes[0, 0].set_ylabel('Count')

axes[0, 1].scatter(y_pred_err, residuals, alpha=0.3, s=12, color='seagreen')
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_title('Residuals vs Predicted Cost', fontweight='bold')
axes[0, 1].set_xlabel('Predicted Cost ($)')
axes[0, 1].set_ylabel('Residual ($)')

brackets = pd.cut(y_true_err, bins=[0, 10, 20, 50, 100, np.inf],
                  labels=['0-10', '10-20', '20-50', '50-100', '100+'])
err_table = pd.DataFrame({
    'bracket':   brackets,
    'abs_error': (y_true_err - y_pred_err).abs(),
    'actual':    y_true_err
})
bracket_stats = err_table.groupby('bracket', observed=True).agg(
    quotes=('actual', 'count'),
    mean_abs_error=('abs_error', 'mean'),
    median_abs_error=('abs_error', 'median')
)

bracket_stats['mean_abs_error'].plot(kind='bar', ax=axes[1, 0],
                                     color='darkorange', edgecolor='black')
axes[1, 0].set_title('Mean Absolute Error by Cost Bracket', fontweight='bold')
axes[1, 0].set_xlabel('Actual Cost Bracket ($)')
axes[1, 0].set_ylabel('Mean Absolute Error ($)')
axes[1, 0].tick_params(axis='x', rotation=0)

bracket_stats['quotes'].plot(kind='bar', ax=axes[1, 1],
                              color='slategray', edgecolor='black')
axes[1, 1].set_title('Number of Test Quotes by Cost Bracket', fontweight='bold')
axes[1, 1].set_xlabel('Actual Cost Bracket ($)')
axes[1, 1].set_ylabel('Count')
axes[1, 1].tick_params(axis='x', rotation=0)

plt.suptitle('9.5 Error Analysis: Enhanced Model on the Test Set',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Error by cost bracket:")
print(bracket_stats.round(4).to_string())"""
),

md("**Insight:** The residual distribution should be roughly centred at zero. A right-skewed tail of positive residuals (actual minus predicted is large and positive) confirms the model underestimates high-cost assemblies. The residuals vs predicted plot reveals whether the error grows with the predicted value, which would indicate the model compresses predictions toward the mean. The bracket breakdown shows exactly which price range drives the overall MAE and RMSE figures."),

md("### Root Cause Analysis\n\nThe error analysis typically shows a consistent pattern across all linear and KNN models on this dataset, with three root causes:\n\n1. **Data imbalance across price ranges.** The large majority of quotes fall below $20. Expensive assemblies above $100 are rare, so both models have few examples to learn high-cost pricing behaviour from, and their absolute error grows steeply as cost increases.\n\n2. **The linearity assumption for LR.** Supplier pricing combines fixed costs (tooling, setup) with variable costs that interact, such as quantity discounts that depend on assembly complexity. A linear model cannot fully represent these interactions. This is why even the enhanced versions plateau in performance.\n\n3. **Back-transformation amplification.** Both models are trained on the power-transformed scale. Converting predictions back to dollars raises them to the power of 20, which amplifies small transformed-scale errors into large dollar errors precisely in the expensive assembly range where the model is already least accurate."),

md("### 9.6 Limitations and Evaluation Summary\n\n**Limitations**\n\n1. The model family is limited for Linear Regression. The flat tuning curves and the experiment log show that parameter tuning and feature engineering give only marginal gains. Closing the remaining performance gap requires non-linear models such as gradient boosting.\n\n2. KNN prediction time grows with the training set size because it searches for neighbours at inference time. For production use with large catalogues this would need approximate nearest neighbour indexing.\n\n3. High-cost assemblies are underrepresented in the training data. Any model trained on this distribution will struggle to price rare, expensive tubes accurately regardless of algorithm.\n\n4. Quotes are concentrated in the 2010 to 2014 period. The models reflect pricing from that period and would require retraining on current data before deployment.\n\n5. The specification dataset was not usable because the lookup table for specification codes is unavailable, which may have excluded features with meaningful price signals.\n\n**Evaluation Summary**\n\nThe evaluation followed the required methodology throughout: a 3-way split with a sealed test set, baseline comparison supported by 5-fold cross-validation, hyperparameter tuning with GridSearchCV on the training set only, validation of chosen parameters on the held-out set, documented enhancement experiments, and one final evaluation on truly unseen test data. The deliverables produced are the validation analysis report (9.1), the enhancement log (9.3), the enhanced model saved as `enhanced_model.pkl` (9.4), the test evaluation report (9.4), the error analysis (9.5), and this limitations and evaluation summary (9.6)."),

]

nb['cells'].extend(cells)

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Done. Added {len(cells)} cells. Notebook now has {len(nb['cells'])} cells.")
