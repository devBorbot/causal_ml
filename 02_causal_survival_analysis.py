"""
Causal Survival Analysis of B2B Discount Campaign

v2: Time-to-event outcomes with censoring using scikit-survival

Methods:
1. Propensity Score Weighting (IPW) - for survival ATE
2. T-Learner with Random Survival Forests - separate models for treatment/control
3. Survival Uplift Estimation - difference in survival probabilities at horizon
4. Time-dependent heterogeneous effects - who benefits most at which time points

Goal: Estimate causal effect on time-to-churn and identify "foot-out-the-door" 
      customers where discounts most extend survival (retention).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Survival analysis imports
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import KaplanMeierFitter

# Set style
sns.set_style("whitegrid")

print("=" * 70)
print("CAUSAL SURVIVAL ANALYSIS: B2B DISCOUNT CAMPAIGN")
print("=" * 70)

# ====================
# 1. LOAD AND PREPARE DATA
# ====================

print("\n[1/6] Loading and preparing survival data...")

# NOTE: This expects a CSV with time-to-event + censoring columns
# If you're starting from the original data, you'll need to engineer:
#   - time_to_churn: days from start to churn or last observation
#   - churn_event: 1 if churned, 0 if censored (still active)

df = pd.read_csv('b2b_customer_data_survival.csv')

# Encode categorical variables
le = LabelEncoder()
df['industry_encoded'] = le.fit_transform(df['industry'])

# Define features (X), treatment (T), time, and event
feature_cols = ['contract_value', 'company_size', 'usage_score', 'nps_score',
                'tenure_months', 'support_tickets', 'industry_encoded']

X = df[feature_cols].values
T = df['discount_offered'].values  # Treatment (1=discount, 0=no discount)
time = df['time_to_churn'].values  # Time to event or censoring
event = df['churn_event'].values.astype(bool)  # 1=event occurred, 0=censored

# Create structured survival array for scikit-survival
y_surv = Surv.from_arrays(event, time)

print(f"✓ Data shape: X={X.shape}, T={T.shape}")
print(f"  Treatment rate: {T.mean():.1%}")
print(f"  Event rate (observed churns): {event.mean():.1%}")
print(f"  Censoring rate: {(~event).mean():.1%}")
print(f"  Median time-to-event/censoring: {np.median(time):.1f} days")

# Split into train/test
X_train, X_test, T_train, T_test, y_train, y_test, time_train, time_test, event_train, event_test = train_test_split(
    X, T, y_surv, time, event, test_size=0.3, random_state=42, stratify=T
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ====================
# 2. INVERSE PROPENSITY WEIGHTING (IPW) FOR SURVIVAL
# ====================

print("\n[2/6] Computing propensity scores for IPW...")
print("-" * 70)

# Estimate propensity scores
prop_model = LogisticRegression(max_iter=1000, random_state=42)
prop_model.fit(X_train, T_train)
propensity_scores = prop_model.predict_proba(X_test)[:, 1]
propensity_scores = np.clip(propensity_scores, 0.05, 0.95)

# Calculate IPW weights
weights = np.where(T_test == 1,
                   1 / propensity_scores,
                   1 / (1 - propensity_scores))

print(f"✓ Propensity scores computed")
print(f"  Mean propensity (test): {propensity_scores.mean():.3f}")
print(f"  IPW weight range: [{weights.min():.2f}, {weights.max():.2f}]")

# ====================
# 3. T-LEARNER WITH RANDOM SURVIVAL FORESTS
# ====================

print("\n[3/6] Training Random Survival Forests (T-Learner approach)...")
print("-" * 70)

def make_rsf():
    """Create Random Survival Forest pipeline with scaling"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('rsf', RandomSurvivalForest(
            n_estimators=200,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        ))
    ])

# Split training data by treatment
X_train_treated = X_train[T_train == 1]
y_train_treated = y_train[T_train == 1]

X_train_control = X_train[T_train == 0]
y_train_control = y_train[T_train == 0]

# Fit separate models
print(f"  Training treated model (n={len(X_train_treated)})...")
rsf_treated = make_rsf()
rsf_treated.fit(X_train_treated, y_train_treated)

print(f"  Training control model (n={len(X_train_control)})...")
rsf_control = make_rsf()
rsf_control.fit(X_train_control, y_train_control)

print("✓ Random Survival Forests trained")

# ====================
# 4. SURVIVAL UPLIFT AT MULTIPLE HORIZONS
# ====================

print("\n[4/6] Computing survival uplift at multiple time horizons...")
print("-" * 70)

# Get the trained RSF models from pipelines
rsf_treated_model = rsf_treated.named_steps['rsf']
rsf_control_model = rsf_control.named_steps['rsf']

# Scale test data
X_test_scaled_t = rsf_treated.named_steps['scaler'].transform(X_test)
X_test_scaled_c = rsf_control.named_steps['scaler'].transform(X_test)

# Predict survival functions
surv_funcs_treated = rsf_treated_model.predict_survival_function(X_test_scaled_t)
surv_funcs_control = rsf_control_model.predict_survival_function(X_test_scaled_c)

def survival_at_horizon(surv_funcs, t_star):
    """Extract survival probability at time horizon t_star from survival functions"""
    probs = []
    for fn in surv_funcs:
        times_arr = fn.x
        surv_arr = fn.y
        if t_star >= times_arr[-1]:
            probs.append(float(surv_arr[-1]))
        else:
            probs.append(float(np.interp(t_star, times_arr, surv_arr)))
    return np.array(probs)

# Compute uplift at multiple horizons (90, 180, 365 days)
horizons = [90, 180, 365]
uplift_results = {}

for horizon in horizons:
    surv_prob_treated = survival_at_horizon(surv_funcs_treated, horizon)
    surv_prob_control = survival_at_horizon(surv_funcs_control, horizon)
    
    # Uplift = difference in survival probability (higher = better for retention)
    uplift = surv_prob_treated - surv_prob_control
    uplift_results[horizon] = {
        'uplift': uplift,
        'surv_treated': surv_prob_treated,
        'surv_control': surv_prob_control
    }
    
    print(f"\nHorizon: {horizon} days")
    print(f"  Mean survival prob (treated):  {surv_prob_treated.mean():.3f}")
    print(f"  Mean survival prob (control):  {surv_prob_control.mean():.3f}")
    print(f"  Mean uplift: {uplift.mean():.4f} ({uplift.mean()*100:.2f} pp)")
    print(f"  Uplift range: [{uplift.min():.4f}, {uplift.max():.4f}]")
    print(f"  Interpretation: Discount {'increases' if uplift.mean() > 0 else 'decreases'} "
          f"retention probability by {abs(uplift.mean())*100:.1f} pp at {horizon} days")

# Use 365-day horizon as primary for segmentation
primary_horizon = 365
uplift_primary = uplift_results[primary_horizon]['uplift']

# ====================
# 5. KAPLAN-MEIER COMPARISON
# ====================

print("\n[5/6] Computing Kaplan-Meier survival curves...")
print("-" * 70)

# Fit KM curves for treated and control in test set
kmf_treated = KaplanMeierFitter()
kmf_control = KaplanMeierFitter()

kmf_treated.fit(time_test[T_test == 1], event_test[T_test == 1], label='Treated (Discount)')
kmf_control.fit(time_test[T_test == 0], event_test[T_test == 0], label='Control (No Discount)')

print(f"✓ Kaplan-Meier curves fitted")
print(f"  Median survival (treated): {kmf_treated.median_survival_time_:.1f} days")
print(f"  Median survival (control): {kmf_control.median_survival_time_:.1f} days")

# ====================
# 6. VISUALIZE RESULTS
# ====================

print("\n[6/6] Creating visualizations...")
print("-" * 70)

# Create test dataframe with predictions
df_test = df.iloc[len(X_train):].reset_index(drop=True).copy()
df_test['uplift_90d'] = uplift_results[90]['uplift']
df_test['uplift_180d'] = uplift_results[180]['uplift']
df_test['uplift_365d'] = uplift_results[365]['uplift']
df_test['surv_prob_treated_365d'] = uplift_results[365]['surv_treated']
df_test['surv_prob_control_365d'] = uplift_results[365]['surv_control']

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: Kaplan-Meier curves
ax1 = fig.add_subplot(gs[0, 0])
kmf_treated.plot_survival_function(ax=ax1, ci_show=True)
kmf_control.plot_survival_function(ax=ax1, ci_show=True)
ax1.set_title('Kaplan-Meier Survival Curves', fontsize=12, fontweight='bold')
ax1.set_xlabel('Days')
ax1.set_ylabel('Survival Probability')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Uplift distribution at 365 days
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(uplift_primary, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
ax2.axvline(uplift_primary.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {uplift_primary.mean():.3f}')
ax2.axvline(0, color='black', linestyle='-', linewidth=1)
ax2.set_title(f'Survival Uplift Distribution ({primary_horizon} days)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Uplift (Δ Survival Probability)')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Uplift across horizons (boxplot)
ax3 = fig.add_subplot(gs[0, 2])
uplift_data = [uplift_results[h]['uplift'] for h in horizons]
bp = ax3.boxplot(uplift_data, labels=[f'{h}d' for h in horizons], patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
ax3.axhline(0, color='red', linestyle='--', linewidth=1)
ax3.set_title('Uplift Across Time Horizons', fontsize=12, fontweight='bold')
ax3.set_xlabel('Time Horizon')
ax3.set_ylabel('Uplift')
ax3.grid(True, alpha=0.3)

# Plot 4: Uplift by contract value
ax4 = fig.add_subplot(gs[1, 0])
scatter = ax4.scatter(df_test['contract_value'], uplift_primary,
                     c=uplift_primary, cmap='RdYlGn', alpha=0.6, s=30)
ax4.axhline(0, color='black', linestyle='-', linewidth=1)
ax4.set_title('Uplift by Contract Value', fontsize=12, fontweight='bold')
ax4.set_xlabel('Contract Value ($K)')
ax4.set_ylabel('Uplift (365 days)')
plt.colorbar(scatter, ax=ax4, label='Uplift')
ax4.grid(True, alpha=0.3)

# Plot 5: Uplift by company size
ax5 = fig.add_subplot(gs[1, 1])
scatter = ax5.scatter(df_test['company_size'], uplift_primary,
                     c=uplift_primary, cmap='RdYlGn', alpha=0.6, s=30)
ax5.axhline(0, color='black', linestyle='-', linewidth=1)
ax5.set_xscale('log')
ax5.set_title('Uplift by Company Size', fontsize=12, fontweight='bold')
ax5.set_xlabel('Company Size (log scale)')
ax5.set_ylabel('Uplift (365 days)')
plt.colorbar(scatter, ax=ax5, label='Uplift')
ax5.grid(True, alpha=0.3)

# Plot 6: Uplift by usage score
ax6 = fig.add_subplot(gs[1, 2])
scatter = ax6.scatter(df_test['usage_score'], uplift_primary,
                     c=uplift_primary, cmap='RdYlGn', alpha=0.6, s=30)
ax6.axhline(0, color='black', linestyle='-', linewidth=1)
ax6.set_title('Uplift by Usage Score', fontsize=12, fontweight='bold')
ax6.set_xlabel('Usage Score')
ax6.set_ylabel('Uplift (365 days)')
plt.colorbar(scatter, ax=ax6, label='Uplift')
ax6.grid(True, alpha=0.3)

# Plot 7: Segment summary (foot-out-the-door analysis)
ax7 = fig.add_subplot(gs[2, 0])
df_test['uplift_segment'] = pd.cut(uplift_primary, 
                                    bins=[-np.inf, -0.05, 0, 0.05, np.inf],
                                    labels=['Strong Negative', 'Weak Negative', 
                                           'Weak Positive', 'Strong Positive'])
segment_counts = df_test['uplift_segment'].value_counts().sort_index()
colors = ['darkred', 'lightcoral', 'lightgreen', 'darkgreen']
segment_counts.plot(kind='bar', ax=ax7, color=colors)
ax7.set_title('Customer Segments by Uplift', fontsize=12, fontweight='bold')
ax7.set_xlabel('Uplift Segment')
ax7.set_ylabel('Number of Customers')
ax7.set_xticklabels(ax7.get_xticklabels(), rotation=45, ha='right')
ax7.grid(True, alpha=0.3)

# Plot 8: Survival probability by uplift quartile
ax8 = fig.add_subplot(gs[2, 1])
df_test['uplift_quartile'] = pd.qcut(uplift_primary, q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
for quartile in df_test['uplift_quartile'].unique():
    mask = df_test['uplift_quartile'] == quartile
    kmf_temp = KaplanMeierFitter()
    kmf_temp.fit(time_test[mask], event_test[mask], label=str(quartile))
    kmf_temp.plot_survival_function(ax=ax8, ci_show=False)
ax8.set_title('Survival by Uplift Quartile', fontsize=12, fontweight='bold')
ax8.set_xlabel('Days')
ax8.set_ylabel('Survival Probability')
ax8.legend(loc='best')
ax8.grid(True, alpha=0.3)

# Plot 9: Treated vs Control survival by uplift segment
ax9 = fig.add_subplot(gs[2, 2])
high_uplift_mask = uplift_primary > 0.05
x_pos = np.arange(2)
treated_surv = [df_test.loc[~high_uplift_mask, 'surv_prob_treated_365d'].mean(),
                df_test.loc[high_uplift_mask, 'surv_prob_treated_365d'].mean()]
control_surv = [df_test.loc[~high_uplift_mask, 'surv_prob_control_365d'].mean(),
                df_test.loc[high_uplift_mask, 'surv_prob_control_365d'].mean()]
width = 0.35
ax9.bar(x_pos - width/2, control_surv, width, label='Control', color='coral', alpha=0.8)
ax9.bar(x_pos + width/2, treated_surv, width, label='Treated', color='steelblue', alpha=0.8)
ax9.set_title('365-day Survival: Low vs High Uplift', fontsize=12, fontweight='bold')
ax9.set_ylabel('Survival Probability')
ax9.set_xticks(x_pos)
ax9.set_xticklabels(['Low Uplift\n(≤0.05)', 'High Uplift\n(>0.05)'])
ax9.legend()
ax9.grid(True, alpha=0.3, axis='y')

plt.suptitle('Causal Survival Analysis: Treatment Effect Heterogeneity', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('causal_survival_analysis_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: causal_survival_analysis_results.png")

# ====================
# 7. BUSINESS RECOMMENDATIONS
# ====================

print("\n" + "=" * 70)
print("BUSINESS RECOMMENDATIONS: FOOT-OUT-THE-DOOR RETENTION")
print("=" * 70)

# Identify high-uplift customers (strong positive response)
high_uplift_customers = df_test[df_test['uplift_365d'] > 0.05]
low_or_negative = df_test[df_test['uplift_365d'] <= 0]

# Identify "foot-out-the-door" customers: low baseline survival + high uplift
foot_out_door = df_test[
    (df_test['surv_prob_control_365d'] < 0.5) &  # High risk without intervention
    (df_test['uplift_365d'] > 0.05)  # But responds well to discount
]

print(f"\n1. TARGETING STRATEGY:")
print(f"   - {len(high_uplift_customers)} customers ({len(high_uplift_customers)/len(df_test)*100:.1f}%) "
      f"show strong positive uplift")
print(f"   - {len(low_or_negative)} customers ({len(low_or_negative)/len(df_test)*100:.1f}%) "
      f"show no/negative response")
print(f"   - Focus retention discounts on high-uplift segment")

print(f"\n2. FOOT-OUT-THE-DOOR SEGMENT (Critical for retention):")
print(f"   - {len(foot_out_door)} customers ({len(foot_out_door)/len(df_test)*100:.1f}%) "
      f"are high-risk BUT high-response")
print(f"   - Baseline 365-day survival: {foot_out_door['surv_prob_control_365d'].mean():.1%}")
print(f"   - With discount: {foot_out_door['surv_prob_treated_365d'].mean():.1%}")
print(f"   - Average uplift: {foot_out_door['uplift_365d'].mean():.1%} increase in retention")

print(f"\n3. HIGH-UPLIFT CUSTOMER PROFILE:")
if len(high_uplift_customers) > 0 and len(low_or_negative) > 0:
    print(f"   - Avg contract value: ${high_uplift_customers['contract_value'].mean():.0f}K "
          f"(vs ${low_or_negative['contract_value'].mean():.0f}K)")
    print(f"   - Avg company size: {high_uplift_customers['company_size'].mean():.0f} "
          f"(vs {low_or_negative['company_size'].mean():.0f})")
    print(f"   - Avg usage score: {high_uplift_customers['usage_score'].mean():.1f} "
          f"(vs {low_or_negative['usage_score'].mean():.1f})")
    print(f"   - Avg NPS: {high_uplift_customers['nps_score'].mean():.1f} "
          f"(vs {low_or_negative['nps_score'].mean():.1f})")

print(f"\n4. EXPECTED RETENTION IMPACT:")
baseline_365d = df_test['surv_prob_control_365d'].mean()
treated_365d_high_uplift = high_uplift_customers['surv_prob_treated_365d'].mean()
print(f"   - Overall baseline 365-day retention: {baseline_365d:.1%}")
print(f"   - High-uplift segment with discount: {treated_365d_high_uplift:.1%}")
print(f"   - Expected customers retained (high-uplift): "
      f"{len(high_uplift_customers) * high_uplift_customers['uplift_365d'].mean():.0f}")

print(f"\n5. TIME-SENSITIVE INSIGHTS:")
print(f"   - 90-day uplift: {uplift_results[90]['uplift'].mean()*100:.2f} pp")
print(f"   - 180-day uplift: {uplift_results[180]['uplift'].mean()*100:.2f} pp")
print(f"   - 365-day uplift: {uplift_results[365]['uplift'].mean()*100:.2f} pp")
print(f"   → Consider timing interventions based on risk horizon")

print(f"\n6. ACTION ITEMS:")
print(f"   ✓ Deploy retention discounts to 'foot-out-the-door' segment FIRST")
print(f"   ✓ Monitor high-risk customers (low baseline survival) approaching 90-180 day mark")
print(f"   ✓ Avoid discounting low/negative uplift customers (margin waste)")
print(f"   ✓ Create early warning system based on survival probability thresholds")
print(f"   ✓ Retrain models quarterly with updated survival data")
print(f"   ✓ A/B test timing: offer at 60d vs 90d vs when survival drops below 70%")

print("\n" + "=" * 70)
print("SURVIVAL ANALYSIS COMPLETE!")
print("=" * 70)
print("\nKEY TAKEAWAY:")
print("Use survival uplift scores to identify customers where discounts")
print("meaningfully extend retention time, especially those at high risk")
print("of imminent churn ('foot-out-the-door'). Time your interventions")
print("strategically based on predicted survival curves.")
print("=" * 70)
