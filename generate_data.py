"""
Generate synthetic data for B2B Sales Discount Campaign Analysis

Use Case: A B2B SaaS company wants to know if offering discounts to at-risk 
customers reduces churn. They need to understand:
1. Does the discount campaign work overall?
2. Which customer segments benefit most from discounts?
3. Should they target all customers or specific segments?

This is a classic uplift modeling problem with confounding variables.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)

def generate_b2b_customer_data(n_samples=5000):
    """
    Generate synthetic B2B customer data with realistic confounding.
    
    Confounders:
    - Sales team tends to offer discounts to larger, more engaged customers
    - These customers might have lower churn regardless of discount
    - Creates selection bias that naive analysis would miss
    """
    
    # Customer characteristics (features)
    data = {}
    
    # Company size (employees) - log-normal distribution
    data['company_size'] = np.random.lognormal(mean=4, sigma=1.5, size=n_samples).astype(int)
    data['company_size'] = np.clip(data['company_size'], 10, 10000)
    
    # Contract value (ARR in thousands)
    # Larger companies tend to have higher contract values
    data['contract_value'] = (data['company_size'] * 0.5 + 
                              np.random.gamma(shape=2, scale=10, size=n_samples))
    data['contract_value'] = np.clip(data['contract_value'], 5, 500)
    
    # Months as customer (tenure)
    data['tenure_months'] = np.random.gamma(shape=2, scale=12, size=n_samples).astype(int)
    data['tenure_months'] = np.clip(data['tenure_months'], 1, 60)
    
    # Support tickets (last 3 months) - more for larger companies
    data['support_tickets'] = np.random.poisson(
        lam=1 + np.log1p(data['company_size']) * 0.3, 
        size=n_samples
    )
    
    # Product usage score (0-100) - engagement metric
    # More engaged customers use product more
    data['usage_score'] = np.random.beta(a=5, b=2, size=n_samples) * 100
    
    # NPS score (-100 to 100) - customer satisfaction
    # Correlated with usage and tenure
    data['nps_score'] = (
        data['usage_score'] * 0.4 + 
        np.sqrt(data['tenure_months']) * 3 +
        np.random.normal(0, 15, n_samples)
    )
    data['nps_score'] = np.clip(data['nps_score'], -100, 100)
    
    # Industry (categorical)
    industries = ['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail']
    industry_probs = [0.3, 0.2, 0.2, 0.15, 0.15]
    data['industry'] = np.random.choice(industries, size=n_samples, p=industry_probs)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # --- TREATMENT ASSIGNMENT (with realistic selection bias) ---
    # Sales team offers discounts based on customer value and churn risk
    # Higher value customers AND those showing churn signals get discounts
    
    # Propensity to receive discount (selection bias)
    propensity_score = stats.logistic.cdf(
        -2.5 +  # Base rate
        0.01 * df['contract_value'] +  # Prefer high-value customers
        -0.02 * df['usage_score'] +  # Target low engagement
        0.005 * df['support_tickets'] +  # Target those with issues
        -0.01 * df['nps_score']  # Target detractors
    )
    
    # Treatment assignment based on propensity
    df['discount_offered'] = (np.random.random(n_samples) < propensity_score).astype(int)
    
    # --- OUTCOME (Churn) ---
    # True causal effect: Discount reduces churn, but effect is heterogeneous
    
    # Base churn probability (without treatment)
    base_churn_logit = (
        -1.0 +  # Baseline
        -0.008 * df['usage_score'] +  # High engagement = lower churn
        -0.015 * df['nps_score'] +  # Promoters don't churn
        -0.01 * np.sqrt(df['tenure_months']) +  # Tenure matters
        0.002 * df['contract_value'] +  # Small companies churn less (proportionally)
        0.05 * df['support_tickets']  # Issues increase churn
    )
    
    # Heterogeneous treatment effect (CATE)
    # Discount works better for:
    # - Mid-size companies (sweet spot)
    # - Those with moderate engagement (recoverable)
    # - Not already loyal customers
    
    treatment_effect = (
        -1.2 +  # Average treatment effect (reduces churn)
        -0.0001 * (df['company_size'] - 200) ** 2 +  # Quadratic: best for mid-size
        0.01 * df['usage_score'] +  # Less effective for engaged users (ceiling effect)
        0.008 * np.abs(df['nps_score'])  # Less effective for extreme NPS
    )
    
    # Final churn probability
    churn_logit = base_churn_logit + treatment_effect * df['discount_offered']
    churn_prob = stats.logistic.cdf(churn_logit)
    
    # Generate binary churn outcome
    df['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Add some noise to make it realistic
    df['churned'] = np.where(
        np.random.random(n_samples) < 0.02,  # 2% random flips
        1 - df['churned'],
        df['churned']
    )
    
    # Store true treatment effect for validation (in practice, unknown)
    df['true_treatment_effect'] = treatment_effect
    
    return df


if __name__ == "__main__":
    # Generate data
    print("Generating B2B customer data...")
    df = generate_b2b_customer_data(n_samples=5000)
    
    # Save to CSV
    df.to_csv('b2b_customer_data.csv', index=False)
    print(f"✓ Generated {len(df)} customer records")
    
    # Basic statistics
    print(f"\nTreatment rate: {df['discount_offered'].mean():.1%}")
    print(f"Overall churn rate: {df['churned'].mean():.1%}")
    print(f"Churn rate (discount): {df[df['discount_offered']==1]['churned'].mean():.1%}")
    print(f"Churn rate (no discount): {df[df['discount_offered']==0]['churned'].mean():.1%}")
    print(f"Naive effect: {(df[df['discount_offered']==1]['churned'].mean() - df[df['discount_offered']==0]['churned'].mean())*100:.1f} pp")
    print(f"True ATE: {df['true_treatment_effect'].mean():.3f} (logit scale)")
