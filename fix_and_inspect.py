import json, re

NB = r"c:\Users\nweke\OneDrive\Desktop\Industrial_Analytics\industrial-analytics-ss2026\SS2026_IBS6_IA_GR04_Chinelo_Nweke_Machine_Learning.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# ── 1. REMOVE EM DASHES ─────────────────────────────────────────
em_dash = "—"
fixed_cells = 0
fixed_lines = 0

for cell in nb["cells"]:
    changed = False
    new_source = []
    for line in cell["source"]:
        if em_dash in line:
            line = line.replace(em_dash, "-")
            fixed_lines += 1
            changed = True
        new_source.append(line)
    if changed:
        cell["source"] = new_source
        fixed_cells += 1

print(f"Em dashes fixed: {fixed_lines} lines across {fixed_cells} cells\n")

# ── 2. INSPECT COMPLETENESS ──────────────────────────────────────
print("=" * 55)
print("NOTEBOOK STRUCTURE OVERVIEW")
print("=" * 55)

task8_checks = {
    "Feature analysis (F-regression / MI)":       False,
    "Feature selection (selected_features)":       False,
    "3-way split (train/val/test)":                False,
    "evaluate() helper function":                  False,
    "LR Model A baseline":                         False,
    "LR Model B (power-transformed) baseline":     False,
    "LR 5-fold cross-validation":                  False,
    "LR GridSearchCV Ridge":                       False,
    "LR GridSearchCV Lasso":                       False,
    "LR Validation phase 4":                       False,
    "LR Final test phase 5":                       False,
    "LR Results summary":                          False,
    "Standardised coefficients chart":             False,
    "SHAP LinearExplainer":                        False,
    "KNN baseline":                                False,
    "KNN GridSearchCV tuning":                     False,
    "KNN Validation phase 4":                      False,
    "KNN Final test phase 5":                      False,
    "KNN Results summary":                         False,
    "Final LR vs KNN comparison":                  False,
}

task9_checks = {
    "9.1 Validation analysis report":              False,
    "Weakness identification":                     False,
    "Experiment 1 SelectKBest":                    False,
    "Experiment 2 Polynomial features":            False,
    "Experiment 3 Manhattan distance":             False,
    "Enhancement comparison log":                  False,
    "9.4 Final test enhanced model":               False,
    "Error analysis by cost bracket":              False,
    "Residual plot":                               False,
}

for cell in nb["cells"]:
    src = "".join(cell["source"])

    # Task 8
    if "f_regression" in src and "mutual_info" in src:
        task8_checks["Feature analysis (F-regression / MI)"] = True
    if "selected_features" in src and "diameter" in src:
        task8_checks["Feature selection (selected_features)"] = True
    if "X_trainval" in src and "X_test" in src and "test_size=0.2" in src:
        task8_checks["3-way split (train/val/test)"] = True
    if "def evaluate" in src:
        task8_checks["evaluate() helper function"] = True
    if "model_A" in src and "y_train_raw" in src:
        task8_checks["LR Model A baseline"] = True
    if "model_B" in src and "LinearRegression" in src:
        task8_checks["LR Model B (power-transformed) baseline"] = True
    if "cross_validate" in src and "LinearRegression" in src:
        task8_checks["LR 5-fold cross-validation"] = True
    if "ridge_gs" in src or ("Ridge" in src and "GridSearchCV" in src):
        task8_checks["LR GridSearchCV Ridge"] = True
    if "lasso_gs" in src or ("Lasso" in src and "GridSearchCV" in src):
        task8_checks["LR GridSearchCV Lasso"] = True
    if "PHASE 4" in src or ("res_ridge" in src and "res_lasso" in src):
        task8_checks["LR Validation phase 4"] = True
    if "final_lr_pipeline" in src and "X_test" in src:
        task8_checks["LR Final test phase 5"] = True
    if "summary_lr" in src or ("Phase 5" in src and "UNSEEN TEST SET" in src and "LinearRegression" in src):
        task8_checks["LR Results summary"] = True
    if "std_coef" in src or ("coef_" in src and "barh" in src):
        task8_checks["Standardised coefficients chart"] = True
    if "shap" in src.lower() and "LinearExplainer" in src:
        task8_checks["SHAP LinearExplainer"] = True
    if "knn_base" in src and "KNeighborsRegressor" in src:
        task8_checks["KNN baseline"] = True
    if "knn_gs" in src and "GridSearchCV" in src:
        task8_checks["KNN GridSearchCV tuning"] = True
    if "knn_gs.best_estimator_" in src and "X_val" in src:
        task8_checks["KNN Validation phase 4"] = True
    if "final_knn_pipeline" in src and "X_test" in src:
        task8_checks["KNN Final test phase 5"] = True
    if "summary_knn" in src:
        task8_checks["KNN Results summary"] = True
    if "comparison" in src and "res_lr_final" in src and "res_knn_final" in src:
        task8_checks["Final LR vs KNN comparison"] = True

    # Task 9
    if "9.1" in src or "VALIDATION ANALYSIS REPORT" in src:
        task9_checks["9.1 Validation analysis report"] = True
    if "weakness" in src.lower() or "Identified weaknesses" in src:
        task9_checks["Weakness identification"] = True
    if "SelectKBest" in src and "exp1" in src:
        task9_checks["Experiment 1 SelectKBest"] = True
    if "PolynomialFeatures" in src and "exp2" in src:
        task9_checks["Experiment 2 Polynomial features"] = True
    if "manhattan" in src.lower() and "exp3" in src:
        task9_checks["Experiment 3 Manhattan distance"] = True
    if "exp_log" in src or "ENHANCEMENT LOG" in src:
        task9_checks["Enhancement comparison log"] = True
    if "9.4" in src or "FINAL TEST EVALUATION OF THE ENHANCED" in src:
        task9_checks["9.4 Final test enhanced model"] = True
    if "bracket" in src and "mean_abs_error" in src:
        task9_checks["Error analysis by cost bracket"] = True
    if "residual" in src.lower() and "plt" in src:
        task9_checks["Residual plot"] = True

print("\nTASK 8 CHECKLIST:")
for k, v in task8_checks.items():
    status = "OK" if v else "MISSING"
    print(f"  [{status:7s}]  {k}")

print("\nTASK 9 CHECKLIST:")
for k, v in task9_checks.items():
    status = "OK" if v else "MISSING"
    print(f"  [{status:7s}]  {k}")

t8_ok  = sum(task8_checks.values())
t9_ok  = sum(task9_checks.values())
print(f"\nTask 8: {t8_ok}/{len(task8_checks)} complete")
print(f"Task 9: {t9_ok}/{len(task9_checks)} complete")

# ── 3. SAVE ──────────────────────────────────────────────────────
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nNotebook saved.")
