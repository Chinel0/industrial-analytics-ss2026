# Supply Management Analytics: Supplier Pricing in the Caterpillar Supply Chain

**Course:** Industrial Analytics, Summer Semester 2026
**Group:** 4
**Milestone Presentation:** 07 May 2026
**Final Presentation:** 25 June 2026

---

## Project Overview

This project addresses a core procurement challenge faced by Caterpillar Inc.: predicting the quoted price of tube assemblies from suppliers. Tube assemblies are multi-component products used in pneumatic systems. They consist of bends, fittings, bosses, brackets, and end connections. The complexity of their geometry, materials, and components causes significant variation in supplier pricing, making it difficult for procurement teams to benchmark quotes or detect overpricing.

The goal is to build a predictive model that forecasts the supplier quote price for a given tube assembly. An accurate model helps procurement teams compare supplier bids objectively, identify outlier quotes, and make faster, better-informed sourcing decisions.

The project follows the CRISP-DM methodology across six phases: Business Understanding, Data Understanding, Data Preparation, Modelling, Evaluation, and Deployment.

---

## Team

| Name | Role |
|---|---|
| Mohammad Sharif Azimy | Project Manager and Data Engineer |
| Antonios Stefanakis | Data Analyst |
| Chinelo Lydia Nweke | Data Engineer |
| Takal Ebrahimkhel | Data Scientist |

---

## Repository Structure

```
industrial-analytics-ss2026/
├── IA_Supply_management.ipynb        <- Main project notebook
├── SS2026_IA_Group4_Milestone_Presentation.pptx
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

The dataset originates from the Caterpillar Tube Assembly Pricing challenge. It consists of multiple relational tables that must be joined using `tube_assembly_id` as the primary key.

| File | Description |
|---|---|
| `train_set.csv` | Supplier quotes with cost as the prediction target |
| `tube.csv` | Physical tube specifications (diameter, wall, length, bends, etc.) |
| `bill_of_materials.csv` | Component composition per tube assembly (up to 8 components) |
| `specs.csv` | Specification codes per tube assembly |
| `tube_end_form.csv` | Lookup table mapping end form IDs to forming type |
| `comp_[name].csv` (x11) | Component-level data including weight, one file per component type |

The target variable is `cost`, the quoted supplier price in US dollars.

---

## Tasks

### Task 1: Business Understanding

The business problem is defined as: predict the supplier quote price for a Caterpillar tube assembly. The pricing structure of a quoted assembly follows the form: quoted price = fixed cost + variable cost + margin. Fixed costs include tooling, setup, and special processes. Variable costs include material, labour, and machining. Margin depends on the supplier relationship and risk profile.

Two pricing regimes exist in the dataset: bracket pricing, where price changes across quantity tiers, and non-bracket pricing, where a fixed price applies once the minimum order quantity is reached. Understanding this distinction is critical to feature engineering and model design.

**Business objectives:**
- Forecast supplier quote price at the assembly level.
- Identify the most important cost drivers in the data.
- Support faster and better-informed supplier negotiation.

**Success criteria:**
- Strong prediction accuracy across both pricing regimes.
- Interpretable feature importance showing which factors drive cost.
- Clean data joins, consistent quantity handling, and robust engineered features.

---

### Task 2: Data Collection and Initial Loading

All datasets are loaded using `pandas.read_csv`. The four main tables (train set, tube, bill of materials, specifications) are imported and confirmed to load correctly. The 11 component files are loaded iteratively to build a unified component weight table. Initial inspection confirms column names, data types, index structure, and dataset dimensions before any analysis begins.

---

### Task 3: Dataset Exploration

Each of the four main datasets is explored individually before merging.

**Train Dataset (Supplier Quotes):**
- Contains supplier quotes with fields for tube assembly ID, supplier, quote date, annual usage, minimum order quantity, bracket pricing flag, quantity ordered, and cost.
- Quote dates span approximately 34 years (September 1982 to January 2017), with the vast majority of quotes concentrated in the 2010 to 2014 period. This temporal imbalance is noted for modelling considerations.

**Tube Dataset:**
- Contains physical specifications per tube assembly including diameter, wall thickness, length, bend radius, number of bends, number of bosses, number of brackets, and end connection types.
- Most numeric features are right-skewed, with many assemblies having simple geometries and a long tail of complex configurations.

**Bill of Materials Dataset:**
- Contains up to 8 component slots per assembly, each with a component ID and quantity.
- Most assemblies contain few components. The total quantity distribution is right-skewed with a mean of approximately 2.74 components per assembly.

**Specifications Dataset:**
- Contains specification codes per assembly. No lookup table is available to decode the specification IDs, which limits the usefulness of this dataset for modelling.

---

### Task 4: Data Quality Verification

Data quality is assessed across four dimensions for all datasets.

**Completeness:** Missing values are counted and reported as percentages. The bill of materials and tube datasets contain expected nulls in optional component slots, which are later handled by filling with zero.

**Uniqueness:** Duplicate rows are checked across all datasets. No significant duplicates are found.

**Timeliness:** The quote date range is verified. Quotes span from September 1982 to January 2017, confirming the dataset covers decades of historical pricing data. The heavy concentration in 2010 to 2014 is flagged as a potential source of temporal bias.

**Consistency:** Text columns are inspected for inconsistent casing, extra whitespace, and label variants. The supplier and tube end form columns are checked for normalisation issues. No critical consistency problems are found, though the categorical fields require encoding before modelling.

**Accuracy:** Outliers are identified using the interquartile range method on all numeric columns in the tube and train datasets. Outliers are retained as they represent genuine high-complexity assemblies rather than data entry errors.

---

### Task 5: Data Preparation and Merging

The datasets are merged using `tube_assembly_id` as the primary key to produce a unified modelling dataset.

**Step 1: Train and Tube merge**
The train set and tube dataset are joined on `tube_assembly_id` to create `NewTrain`, combining supplier quote information with physical tube specifications.

**Step 2: Bill of Materials aggregation**
Total component quantity per assembly is computed by summing all quantity columns in the bill of materials. Component IDs in the bill of materials are then replaced with their corresponding weights from the component files. Total component weight per assembly is calculated by multiplying each component weight by its quantity and summing across all component slots.

**Step 3: Final merge**
The bill of materials (with computed Total_quantity and Total_weight) is merged with `NewTrain` on `tube_assembly_id` to produce `Final_DF`, the clean, unified dataset used for all subsequent analysis and modelling.

**Step 4: Cleaning and dropping**
The `quote_date` column is dropped after its components are extracted as features. The `material_id` column is dropped as it provides no useful signal for price prediction. Missing values in forming columns are filled with zero, representing assemblies with no end forming applied.

---

### Task 6: Feature Engineering

Six feature engineering steps are applied to prepare the data for modelling.

**1. Date Feature Extraction**
The `quote_date` column is decomposed into five new numeric columns: year, month, day of year, day of week, and day. This converts a single date field into multiple features capable of capturing seasonal and cyclical pricing patterns.

**2. Categorical Encoding**
Binary text fields (Y/N) such as `bracket_pricing`, `end_a_1x`, `end_a_2x`, `end_x_1x`, and `end_x_2x` are converted to 0 and 1 using `LabelEncoder`. This makes them compatible with numeric modelling algorithms.

**3. End Forming Features**
The `tube_end_form` lookup table is used to extract whether each tube end (end_a and end_x) has forming applied. Two new binary columns, `end_a_forming` and `end_x_forming`, are created. Forming indicates custom shaping at the tube end, which increases manufacturing complexity and typically raises price. Missing values in these columns are filled with zero.

**4. Total Quantity and Total Weight from Bill of Materials**
Total component quantity per assembly is summed from the bill of materials. Total component weight is calculated by mapping component IDs to their weights and multiplying by quantities. Both features are added to `Final_DF` as they are expected to correlate positively with cost.

**5. Identifier Splitting**
The `tube_assembly_id` and `supplier` columns are split on the hyphen separator to extract the numeric portion only. This converts them from string identifiers to integer values suitable for use as features.

**6. Minimum Order Quantity Alignment**
Where `min_order_quantity` is zero or less than the actual `quantity` ordered, it is aligned to match `quantity`. This corrects a logical inconsistency in the data where the minimum order threshold would otherwise appear lower than the quantity actually placed.

---

### Task 7: Descriptive Data Analytics and Visualisation

A series of visualisations is produced to understand the distributions, relationships, and patterns in `Final_DF`.

**Cost Distribution:** The cost variable is severely right-skewed with the majority of quotes concentrated between $0 and $15. The distribution has a sharp peak near $5 to $10 and a long tail extending beyond $100. This skewness motivates the log transformation applied during modelling.

**Correlation Heatmaps:** Two correlation heatmaps are produced. The first uses a masked lower-triangular layout to reduce visual clutter. The second is a full symmetric heatmap. Both confirm that `Total_weight`, `length`, `diameter`, and `quantity` show the strongest positive correlations with cost. Features such as `annual_usage`, `bend_radius`, and `other` show very weak correlations and are considered for removal.

**Supplier Cost Distribution:** A boxplot of cost grouped by supplier reveals significant pricing variability across suppliers. Some suppliers cluster tightly at low cost ranges, indicating predictable pricing. Others show wide interquartile ranges and many high-value outliers, suggesting inconsistent or volume-dependent pricing strategies.

**Pairplot:** A corner pairplot of the key features (diameter, wall, length, num_bends, bend_radius, cost, Total_weight) with KDE diagonals confirms the positive relationship between Total_weight and cost, the right-skewed marginal distributions, and moderate correlations among the physical size features.

**Quotation Period Analysis:** Quotes are grouped into five-year bins. The analysis confirms a sharp concentration of quotes in the 2010 to 2014 period, with very low volumes in earlier decades. This temporal imbalance is noted as a factor that may affect model generalisation.

---

### Task 8: Model Implementation and Evaluation

Task 8 implements a Linear Regression pipeline to predict supplier quote cost. The methodology follows a structured six-phase approach designed to produce validated, trustworthy results rather than a single train-test split.

#### Phase 1: Data Preparation and 3-Way Split

Features are selected and the target variable (`cost`) is log-transformed using `np.log1p` to address the severe right-skew identified in Task 7. The data is divided into three non-overlapping parts: 60% training set, 20% validation set, and 20% test set. The test set is sealed immediately and is not used for any training, tuning, or comparison decision until the final evaluation in Phase 5.

#### Phase 2: Baseline Models

Two baseline models are trained and evaluated on the validation set.

**Model A: Linear Regression on Raw Cost**
A standard Linear Regression pipeline (StandardScaler + LinearRegression) is trained on the untransformed cost values. It is evaluated on the validation set and assessed via 5-fold cross-validation on the training set.

**Model B: Linear Regression on Log-Transformed Cost**
The same pipeline is trained on the log-transformed cost. Predictions are converted back to the original dollar scale using `np.expm1` for evaluation. Both validation metrics and 5-fold cross-validation scores are reported.

The comparison between Model A and Model B demonstrates the value of cross-validation: Model A may appear to perform better on a single validation split, but the 5-fold CV R² reveals that Model B is more stable and consistent across different subsets of the data. Model B is selected as the basis for all further tuning.

#### Phase 3: Hyperparameter Tuning with GridSearchCV

Two regularised variants of linear regression are tuned using `GridSearchCV` with 5-fold cross-validation on the training set. The validation set remains sealed during this entire phase.

**Ridge Regression (L2 regularisation):** Six alpha values are tested (0.01, 0.1, 1, 10, 100, 1000). The best alpha is selected based on lowest CV MAE on the log-transformed target.

**Lasso Regression (L1 regularisation):** Five alpha values are tested (0.0001, 0.001, 0.01, 0.1, 1). The best alpha is selected on the same criterion.

Tuning curves are plotted for both models, showing how CV MAE changes across the alpha range and marking the selected best value.

#### Phase 4: Validation on Held-Out Validation Set

The best Ridge and Lasso configurations identified in Phase 3 are evaluated on the validation set. This confirms whether the parameters found during tuning actually generalise to data the models have not seen. All metrics are computed on the original dollar scale for a fair comparison. The model with the lowest validation MAE is selected as the final model.

#### Phase 5: Final Evaluation on Unseen Test Set

The best model is retrained on the combined training and validation data. It is then evaluated exactly once on the sealed test set. These are the final reported results. The test set was not used for any decision prior to this step, making the reported metrics a genuine measure of generalisation performance. A final 5-fold CV stability check is also run to confirm consistency.

#### Phase 6: Results Summary

A complete results table is produced showing performance at each stage of the pipeline, from the baselines through to the final test evaluation. All metrics (MAE, RMSE, R²) are in the original dollar scale throughout.

**Conclusion:** Linear Regression with log-transformed cost and regularisation provides a meaningful baseline for supplier price prediction. The low R² values across all linear models are consistent with the EDA finding that cost is driven by complex, non-linear interactions among tube geometry, component weight, supplier strategy, and order quantity. This baseline result serves as a benchmark for the ensemble methods (XGBoost by Takal Ebrahimkhel, CatBoost by Mohammad Sharif Azimy) explored by other team members.

#### Explainability: Feature Importance

Two explainability methods are applied to the final model.

**Standardised Coefficients:** The coefficients of the fitted linear model are plotted as a horizontal bar chart. Because inputs are standardised, the coefficients are on a comparable scale and directly indicate each feature's contribution to the predicted log cost. Positive coefficients increase predicted price; negative coefficients decrease it.

**SHAP Values (SHapley Additive Explanations):** A `shap.LinearExplainer` is applied to the final pipeline. A global summary bar plot shows which features have the highest average impact across the validation set. A dot summary plot shows both magnitude and direction of each feature's effect. A waterfall plot is produced for a single prediction to show step-by-step how each feature pushed the prediction above or below the baseline expected cost.

---

## Key Findings

- Cost is heavily right-skewed. Log transformation significantly improves model stability as confirmed by cross-validation.
- Total weight, tube length, and diameter are the strongest predictors of cost identified by both correlation analysis and SHAP values.
- Supplier identity has a notable effect on cost distribution, with significant pricing variability across suppliers for similar assemblies.
- Cross-validation is essential for reliable evaluation. A single train-validation split can be misleading due to the concentration of high-cost outlier assemblies in one split.
- Linear Regression R² values in the range of 0.28 to 0.38 confirm that the price is not fully captured by linear relationships, motivating the use of gradient boosting methods by other team members.

---

## Requirements

Install all dependencies with:

```
pip install -r requirements.txt
```

Key libraries used:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
joblib
```

---

## How to Run

1. Place all dataset CSV files in the same directory referenced by the data loading cells (or update the file paths in the notebook).
2. Open `IA_Supply_management.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells from top to bottom in order. Do not skip cells, as each phase depends on variables defined in earlier cells.

---

## References

Liu, Yang (2022): *Supply Chain Analytics. Concepts, Techniques and Applications.* Cham: Palgrave Macmillan.

Caterpillar Tube Assembly Pricing Dataset (Kaggle Competition).
