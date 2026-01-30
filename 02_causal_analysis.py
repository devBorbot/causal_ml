"""
Causal Analysis of B2B Discount Campaign using EconML

Methods:
1. Propensity Score Weighting (IPW) - estimates ATE
2. S-Learner - simple meta-learner baseline
3. T-Learner - separate models for treatment/control
4. X-Learner - best for heterogeneous effects
5. Causal Forest - non-parametric CATE estimation

Goal: Estimate true causal effect and identify customer segments where discounts work best
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# EconML imports
from econml.metalearners import SLearner, TLearner, XLearner
from econml.dml import CausalForestDML
from econml.sklearn_extensions.linear_model import WeightedLasso

# Set style
sns.set_style("whitegrid")

print("=" * 70)
print("CAUSAL ANALYSIS: B2B DISCOUNT CAMPAIGN")
print("=" * 70)

# ====================
# 1. LOAD AND PREPARE DATA
# ====================
print("\n[1/6] Loading and preparing data...")

df = pd.read_csv('b2b_customer_data.csv')

# Encode categorical variables
le = LabelEncoder()
df['industry_encoded'] = le.fit_transform(df['industry'])

# Define features (X), treatment (T), and outcome (Y)
feature_cols = ['contract_value', 'company_size', 'usage_score', 'nps_score', 
                'tenure_months', 'support_tickets', 'industry_encoded']

X = df[feature_cols].values
T = df['discount_offered'].values  # Treatment (1=discount, 0=no discount)
Y = df['churned'].values  # Outcome (1=churned, 0=retained)

print(f"✓ Data shape: X={X.shape}, T={T.shape}, Y={Y.shape}")
print(f"  Treatment rate: {T.mean():.1%}")
print(f"  Churn rate: {Y.mean():.1%}")

# Split into train/test
X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
    X, T, Y, test_size=0.3, random_state=42, stratify=T
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ====================
# 2. INVERSE PROPENSITY WEIGHTING (IPW)
# ====================
print("\n[2/6] Computing Average Treatment Effect via IPW...")
print("-" * 70)

# Estimate propensity scores (probability of receiving treatment)
prop_model = LogisticRegression(max_iter=1000, random_state=42)
prop_model.fit(X_train, T_train)
propensity_scores = prop_model.predict_proba(X_test)[:, 1]

# Clip propensity scores to avoid extreme weights (common practice)
propensity_scores = np.clip(propensity_scores, 0.05, 0.95)

# Calculate IPW weights
weights = np.where(T_test == 1, 
                   1 / propensity_scores,  # Treated
                   1 / (1 - propensity_scores))  # Control

# Estimate ATE using IPW
ate_treated = np.average(Y_test[T_test == 1], weights=weights[T_test == 1])
ate_control = np.average(Y_test[T_test == 0], weights=weights[T_test == 0])
ate_ipw = ate_treated - ate_control

print(f"Propensity Score IPW Results:")
print(f"  Weighted E[Y|T=1]: {ate_treated:.4f}")
print(f"  Weighted E[Y|T=0]: {ate_control:.4f}")
print(f"  ATE (IPW): {ate_ipw:.4f} ({ate_ipw*100:.2f} percentage points)")
print(f"  Interpretation: Discount {'reduces' if ate_ipw < 0 else 'increases'} churn by {abs(ate_ipw)*100:.1f} pp")

# ====================
# 3. S-LEARNER
# ====================
print("\n[3/6] Training S-Learner (single model)...")
print("-" * 70)

# S-Learner: Single model with treatment as feature
s_learner = SLearner(
    overall_model=GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
)

s_learner.fit(Y_train, T_train, X=X_train)
s_cate = s_learner.effect(X_test)

print(f"S-Learner Results:")
print(f"  Mean CATE: {s_cate.mean():.4f}")
print(f"  Std CATE: {s_cate.std():.4f}")
print(f"  Min CATE: {s_cate.min():.4f}, Max CATE: {s_cate.max():.4f}")

# ====================
# 4. T-LEARNER
# ====================
print("\n[4/6] Training T-Learner (two models)...")
print("-" * 70)

# T-Learner: Separate models for treated and control
t_learner = TLearner(
    models=[
        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    ]
)

t_learner.fit(Y_train, T_train, X=X_train)
t_cate = t_learner.effect(X_test)

print(f"T-Learner Results:")
print(f"  Mean CATE: {t_cate.mean():.4f}")
print(f"  Std CATE: {t_cate.std():.4f}")
print(f"  Min CATE: {t_cate.min():.4f}, Max CATE: {t_cate.max():.4f}")

# ====================
# 5. X-LEARNER (BEST FOR HETEROGENEOUS EFFECTS)
# ====================
print("\n[5/6] Training X-Learner (cross-fitted)...")
print("-" * 70)

# X-Learner: More sophisticated, good when treatment/control sizes differ
x_learner = XLearner(
    models=[
        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    ],
    propensity_model=LogisticRegression(max_iter=1000, random_state=42)
)

x_learner.fit(Y_train, T_train, X=X_train)
x_cate = x_learner.effect(X_test)

print(f"X-Learner Results:")
print(f"  Mean CATE: {x_cate.mean():.4f}")
print(f"  Std CATE: {x_cate.std():.4f}")
print(f"  Min CATE: {x_cate.min():.4f}, Max CATE: {x_cate.max():.4f}")

# ====================
# 6. CAUSAL FOREST (Non-parametric)
# ====================
print("\n[6/6] Training Causal Forest DML...")
print("-" * 70)

# Causal Forest with Double Machine Learning
# This handles confounding automatically
causal_forest = CausalForestDML(
    model_y=GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
    model_t=GradientBoostingClassifier(n_estimators=50, max_depth=4, random_state=42),
    n_estimators=100,
    max_depth=5,
    random_state=42
)

causal_forest.fit(Y_train, T_train, X=X_train)
cf_cate = causal_forest.effect(X_test)

print(f"Causal Forest Results:")
print(f"  Mean CATE: {cf_cate.mean():.4f}")
print(f"  Std CATE: {cf_cate.std():.4f}")
print(f"  Min CATE: {cf_cate.min():.4f}, Max CATE: {cf_cate.max():.4f}")

# ====================
# 7. COMPARE ALL METHODS
# ====================
print("\n" + "=" * 70)
print("SUMMARY: Comparing All Methods")
print("=" * 70)

results_summary = pd.DataFrame({
    'Method': ['IPW', 'S-Learner', 'T-Learner', 'X-Learner', 'Causal Forest'],
    'ATE': [ate_ipw, s_cate.mean(), t_cate.mean(), x_cate.mean(), cf_cate.mean()],
    'Std': [np.nan, s_cate.std(), t_cate.std(), x_cate.std(), cf_cate.std()],
    'Min CATE': [np.nan, s_cate.min(), t_cate.min(), x_cate.min(), cf_cate.min()],
    'Max CATE': [np.nan, s_cate.max(), t_cate.max(), x_cate.max(), cf_cate.max()]
})

print("\n", results_summary.to_string(index=False))

# Compare to naive estimate
naive_ate = Y_test[T_test == 1].mean() - Y_test[T_test == 0].mean()
print(f"\nNaive ATE (biased): {naive_ate:.4f}")
print(f"Causal ATE (X-Learner): {x_cate.mean():.4f}")
print(f"Bias corrected by: {abs(naive_ate - x_cate.mean()):.4f}")

# ====================
# 8. VISUALIZE HETEROGENEOUS EFFECTS
# ====================
print("\n" + "=" * 70)
print("VISUALIZATION: Treatment Effect Heterogeneity")
print("=" * 70)

# Create test dataframe with predictions
df_test = df.iloc[len(X_train):].reset_index(drop=True).copy()
df_test['x_learner_cate'] = x_cate
df_test['cf_cate'] = cf_cate
df_test['t_learner_cate'] = t_cate

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Causal Analysis: Treatment Effect Heterogeneity', fontsize=16, y=1.00)

# Plot 1: CATE distribution (X-Learner)
ax = axes[0, 0]
ax.hist(x_cate, bins=50, alpha=0.7, edgecolor='black')
ax.axvline(x_cate.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {x_cate.mean():.3f}')
ax.set_title('X-Learner: CATE Distribution')
ax.set_xlabel('Treatment Effect (CATE)')
ax.set_ylabel('Frequency')
ax.legend()

# Plot 2: CATE by contract value
ax = axes[0, 1]
scatter = ax.scatter(df_test['contract_value'], x_cate, 
                     c=x_cate, cmap='RdYlGn_r', alpha=0.6, s=20)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('CATE by Contract Value')
ax.set_xlabel('Contract Value ($K)')
ax.set_ylabel('Treatment Effect')
plt.colorbar(scatter, ax=ax, label='CATE')

# Plot 3: CATE by company size
ax = axes[0, 2]
scatter = ax.scatter(df_test['company_size'], x_cate, 
                     c=x_cate, cmap='RdYlGn_r', alpha=0.6, s=20)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_xscale('log')
ax.set_title('CATE by Company Size')
ax.set_xlabel('Company Size (log scale)')
ax.set_ylabel('Treatment Effect')
plt.colorbar(scatter, ax=ax, label='CATE')

# Plot 4: CATE by usage score
ax = axes[1, 0]
scatter = ax.scatter(df_test['usage_score'], x_cate, 
                     c=x_cate, cmap='RdYlGn_r', alpha=0.6, s=20)
ax.axhline(0, color='black', linestyle='-', linewidth=1)
ax.set_title('CATE by Usage Score')
ax.set_xlabel('Usage Score')
ax.set_ylabel('Treatment Effect')
plt.colorbar(scatter, ax=ax, label='CATE')

# Plot 5: Compare methods
ax = axes[1, 1]
methods_data = [s_cate, t_cate, x_cate, cf_cate]
ax.boxplot(methods_data, labels=['S-Learner', 'T-Learner', 'X-Learner', 'Causal Forest'])
ax.axhline(0, color='red', linestyle='--', linewidth=1)
ax.set_title('CATE Estimates: Method Comparison')
ax.set_ylabel('Treatment Effect')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 6: Uplift segments (X-Learner)
ax = axes[1, 2]
df_test['uplift_segment'] = pd.cut(x_cate, bins=[-np.inf, -0.1, 0, 0.1, np.inf],
                                     labels=['Strong Positive', 'Weak Positive', 'Weak Negative', 'Strong Negative'])
segment_counts = df_test['uplift_segment'].value_counts().sort_index()
segment_counts.plot(kind='bar', ax=ax, color=['darkgreen', 'lightgreen', 'lightcoral', 'darkred'])
ax.set_title('Customer Segments by Uplift')
ax.set_xlabel('Uplift Segment')
ax.set_ylabel('Number of Customers')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('causal_analysis_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: causal_analysis_results.png")

# ====================
# 9. BUSINESS RECOMMENDATIONS
# ====================
print("\n" + "=" * 70)
print("BUSINESS RECOMMENDATIONS")
print("=" * 70)

# Segment customers by uplift
df_test['uplift_score'] = x_cate

# Find who benefits most from treatment
high_uplift = df_test[df_test['uplift_score'] < -0.1]  # Negative = reduces churn
low_uplift = df_test[df_test['uplift_score'] > 0]  # Positive = increases churn

print(f"\n1. TARGETING STRATEGY:")
print(f"   - {len(high_uplift)} customers ({len(high_uplift)/len(df_test)*100:.1f}%) show strong positive response")
print(f"   - {len(low_uplift)} customers ({len(low_uplift)/len(df_test)*100:.1f}%) show negative response")
print(f"   - Focus discounts on high-uplift segment")

print(f"\n2. HIGH-UPLIFT CUSTOMER PROFILE:")
print(f"   - Avg contract value: ${high_uplift['contract_value'].mean():.0f}K (vs ${low_uplift['contract_value'].mean():.0f}K)")
print(f"   - Avg company size: {high_uplift['company_size'].mean():.0f} (vs {low_uplift['company_size'].mean():.0f})")
print(f"   - Avg usage score: {high_uplift['usage_score'].mean():.1f} (vs {low_uplift['usage_score'].mean():.1f})")
print(f"   - Avg NPS: {high_uplift['nps_score'].mean():.1f} (vs {low_uplift['nps_score'].mean():.1f})")

print(f"\n3. EXPECTED IMPACT:")
expected_reduction = high_uplift['uplift_score'].mean()
baseline_churn = df_test['churned'].mean()
print(f"   - Baseline churn rate: {baseline_churn*100:.1f}%")
print(f"   - Expected reduction (high-uplift): {abs(expected_reduction)*100:.1f} pp")
print(f"   - Customers saved: {len(high_uplift) * abs(expected_reduction):.0f} out of {len(high_uplift)}")

print(f"\n4. ACTION ITEMS:")
print(f"   ✓ Deploy discounts only to high-uplift segment")
print(f"   ✓ Avoid offering to negative-uplift customers (waste of margin)")
print(f"   ✓ Monitor actual outcomes and retrain models quarterly")
print(f"   ✓ Consider A/B test on borderline segments")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
