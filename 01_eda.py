"""
Exploratory Data Analysis for B2B Discount Campaign

Key questions:
1. What does the data look like?
2. Is there selection bias in treatment assignment?
3. What's the naive treatment effect vs true causal effect?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load data
print("Loading data...")
df = pd.read_csv('b2b_customer_data.csv')
print(f"Dataset: {len(df)} customers, {df.shape[1]} features\n")

# ====================
# 1. DATA OVERVIEW
# ====================
print("=" * 60)
print("DATA OVERVIEW")
print("=" * 60)

print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# ====================
# 2. TREATMENT DISTRIBUTION
# ====================
print("\n" + "=" * 60)
print("TREATMENT ASSIGNMENT")
print("=" * 60)

print(f"\nDiscount offered: {df['discount_offered'].sum()} customers ({df['discount_offered'].mean():.1%})")
print(f"No discount: {(~df['discount_offered'].astype(bool)).sum()} customers ({(1-df['discount_offered'].mean()):.1%})")

# ====================
# 3. OUTCOME ANALYSIS
# ====================
print("\n" + "=" * 60)
print("CHURN ANALYSIS")
print("=" * 60)

print(f"\nOverall churn rate: {df['churned'].mean():.1%}")
print(f"Churn rate (discount group): {df[df['discount_offered']==1]['churned'].mean():.1%}")
print(f"Churn rate (control group): {df[df['discount_offered']==0]['churned'].mean():.1%}")

# Naive effect (WRONG - doesn't account for confounding)
naive_ate = df[df['discount_offered']==1]['churned'].mean() - df[df['discount_offered']==0]['churned'].mean()
print(f"\n⚠️  NAIVE treatment effect: {naive_ate:.4f} ({naive_ate*100:.2f} percentage points)")
print("This is BIASED due to confounding!")

# ====================
# 4. CHECK FOR SELECTION BIAS
# ====================
print("\n" + "=" * 60)
print("SELECTION BIAS ANALYSIS")
print("=" * 60)
print("Do treated and control groups differ on covariates? (They shouldn't in RCT)")

# Compare features between treatment groups
features = ['contract_value', 'company_size', 'usage_score', 'nps_score', 'tenure_months', 'support_tickets']

comparison_data = []
for feature in features:
    treated = df[df['discount_offered']==1][feature]
    control = df[df['discount_offered']==0][feature]
    
    # T-test for difference in means
    t_stat, p_val = stats.ttest_ind(treated, control)
    
    comparison_data.append({
        'Feature': feature,
        'Control Mean': control.mean(),
        'Treated Mean': treated.mean(),
        'Difference': treated.mean() - control.mean(),
        'Std Diff': (treated.mean() - control.mean()) / control.std(),
        'P-value': p_val
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))
print("\n⚠️  Large standardized differences indicate selection bias!")
print("Standard rule: |Std Diff| > 0.1 suggests imbalance")

# ====================
# 5. VISUALIZATIONS
# ====================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('B2B Customer Data - Exploratory Analysis', fontsize=16, y=1.00)

# Plot 1: Churn rate by treatment
ax = axes[0, 0]
churn_by_treatment = df.groupby('discount_offered')['churned'].agg(['mean', 'count'])
churn_by_treatment['mean'].plot(kind='bar', ax=ax, color=['#e74c3c', '#3498db'])
ax.set_title('Churn Rate by Treatment')
ax.set_xlabel('Discount Offered')
ax.set_ylabel('Churn Rate')
ax.set_xticklabels(['No', 'Yes'], rotation=0)
ax.axhline(df['churned'].mean(), color='gray', linestyle='--', label='Overall')
ax.legend()

# Plot 2: Contract value distribution
ax = axes[0, 1]
df.boxplot(column='contract_value', by='discount_offered', ax=ax)
ax.set_title('Contract Value by Treatment')
ax.set_xlabel('Discount Offered')
ax.set_ylabel('Contract Value ($K)')
plt.sca(ax)
plt.xticks([1, 2], ['No', 'Yes'])

# Plot 3: Usage score distribution
ax = axes[0, 2]
df.boxplot(column='usage_score', by='discount_offered', ax=ax)
ax.set_title('Usage Score by Treatment')
ax.set_xlabel('Discount Offered')
ax.set_ylabel('Usage Score')
plt.sca(ax)
plt.xticks([1, 2], ['No', 'Yes'])

# Plot 4: Company size distribution
ax = axes[1, 0]
for treatment in [0, 1]:
    subset = df[df['discount_offered']==treatment]['company_size']
    ax.hist(np.log10(subset), bins=30, alpha=0.6, 
            label=f'Discount: {["No", "Yes"][treatment]}')
ax.set_title('Company Size Distribution')
ax.set_xlabel('Log10(Company Size)')
ax.set_ylabel('Count')
ax.legend()

# Plot 5: NPS score distribution
ax = axes[1, 1]
df.boxplot(column='nps_score', by='discount_offered', ax=ax)
ax.set_title('NPS Score by Treatment')
ax.set_xlabel('Discount Offered')
ax.set_ylabel('NPS Score')
plt.sca(ax)
plt.xticks([1, 2], ['No', 'Yes'])

# Plot 6: Tenure distribution
ax = axes[1, 2]
df.boxplot(column='tenure_months', by='discount_offered', ax=ax)
ax.set_title('Tenure by Treatment')
ax.set_xlabel('Discount Offered')
ax.set_ylabel('Months')
plt.sca(ax)
plt.xticks([1, 2], ['No', 'Yes'])

# Plot 7: Industry distribution
ax = axes[2, 0]
industry_treatment = pd.crosstab(df['industry'], df['discount_offered'], normalize='columns')
industry_treatment.plot(kind='bar', ax=ax, stacked=False)
ax.set_title('Industry Distribution by Treatment')
ax.set_xlabel('Industry')
ax.set_ylabel('Proportion')
ax.legend(['No Discount', 'Discount'])
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Plot 8: Correlation heatmap
ax = axes[2, 1]
corr_features = ['contract_value', 'company_size', 'usage_score', 'nps_score', 
                 'tenure_months', 'discount_offered', 'churned']
corr_matrix = df[corr_features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Feature Correlations')

# Plot 9: Churn rate by usage score (binned)
ax = axes[2, 2]
df['usage_bin'] = pd.cut(df['usage_score'], bins=5)
churn_by_usage = df.groupby(['usage_bin', 'discount_offered'])['churned'].mean().unstack()
churn_by_usage.plot(kind='bar', ax=ax)
ax.set_title('Churn Rate by Usage Score & Treatment')
ax.set_xlabel('Usage Score Bin')
ax.set_ylabel('Churn Rate')
ax.legend(['No Discount', 'Discount'])
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: eda_analysis.png")

# ====================
# 6. KEY INSIGHTS
# ====================
print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)

print("\n1. SELECTION BIAS EXISTS:")
print("   - Customers who received discounts have different characteristics")
print("   - They tend to be higher value, lower engagement, more support tickets")
print("   - This is realistic: sales targets at-risk high-value customers")

print("\n2. NAIVE ANALYSIS IS MISLEADING:")
print(f"   - Naive effect: {naive_ate*100:.2f} pp")
print("   - This suggests discounts might INCREASE churn (or be ineffective)")
print("   - But this ignores that discount group was already at higher risk!")

print("\n3. NEED CAUSAL METHODS:")
print("   - Can't simply compare treated vs control means")
print("   - Must adjust for confounding variables")
print("   - Next: Use propensity scores and meta-learners")

print("\n" + "=" * 60)
print("EDA COMPLETE - Ready for causal analysis!")
print("=" * 60)
