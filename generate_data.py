"""
Generate synthetic data for B2B Sales Discount Campaign Analysis

Use Case:
A B2B SaaS company wants to know if offering discounts to at-risk
customers reduces churn. They need to understand:
1. Does the discount campaign work overall?
2. Which customer segments benefit most from discounts?
3. Should they target all customers or specific segments?

This is a classic uplift modeling problem with confounding variables.

v2: Adds synthetic time-to-churn and censoring to support survival analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats

# Set seed for reproducibility
np.random.seed(42)


def generate_b2b_customer_data(n_samples: int = 5000) -> pd.DataFrame:
    """
    Generate synthetic B2B customer data with realistic confounding.

    Confounders:
    - Sales team tends to offer discounts to larger, more engaged customers.
    - These customers might have lower churn regardless of discount.
    - Creates selection bias that naive analysis would miss.
    """
    # Customer characteristics (features)
    data: dict = {}

    # Company size (employees) - log-normal distribution
    company_size = np.random.lognormal(mean=4, sigma=1.5, size=n_samples).astype(int)
    company_size = np.clip(company_size, 10, 10000)
    data["company_size"] = company_size

    # Contract value (ARR in thousands) - larger companies tend to have higher contract values
    contract_value = company_size * 0.5 + np.random.gamma(shape=2, scale=10, size=n_samples)
    contract_value = np.clip(contract_value, 5, 500)
    data["contract_value"] = contract_value

    # Months as customer (tenure)
    tenure_months = np.random.gamma(shape=2, scale=12, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 60)
    data["tenure_months"] = tenure_months

    # Support tickets (last 3 months) - more for larger companies
    support_tickets = np.random.poisson(
        lam=1 + np.log1p(company_size) * 0.3,
        size=n_samples,
    )
    data["support_tickets"] = support_tickets

    # Product usage score (0-100) - engagement metric
    usage_score = np.random.beta(a=5, b=2, size=n_samples) * 100
    data["usage_score"] = usage_score

    # NPS score (-100 to 100) - correlated with usage and tenure
    nps_score = (
        usage_score * 0.4
        + np.sqrt(tenure_months) * 3
        + np.random.normal(0, 15, n_samples)
    )
    nps_score = np.clip(nps_score, -100, 100)
    data["nps_score"] = nps_score

    # Industry (categorical)
    industries = ["Technology", "Healthcare", "Finance", "Manufacturing", "Retail"]
    industry_probs = [0.3, 0.2, 0.2, 0.15, 0.15]
    industry = np.random.choice(industries, size=n_samples, p=industry_probs)
    data["industry"] = industry

    # Create DataFrame
    df = pd.DataFrame(data)

    # =========================================================
    # TREATMENT ASSIGNMENT (with realistic selection bias)
    # =========================================================
    # Sales team offers discounts based on customer value and churn risk:
    # - Higher value customers
    # - Low engagement / high tickets / detractors

    propensity_score = stats.logistic.cdf(
        -2.5  # Base rate
        + 0.01 * df["contract_value"]  # Prefer high-value customers
        - 0.02 * df["usage_score"]  # Target low engagement
        + 0.005 * df["support_tickets"]  # Target those with issues
        - 0.01 * df["nps_score"]  # Target detractors
    )

    df["discount_offered"] = (np.random.random(n_samples) < propensity_score).astype(int)

    # =========================================================
    # OUTCOME (Binary Churn, as in v1)
    # =========================================================
    # True causal effect: Discount reduces churn, but effect is heterogeneous.

    # Base churn probability (without treatment)
    base_churn_logit = (
        -1.0  # Baseline
        - 0.008 * df["usage_score"]  # High engagement = lower churn
        - 0.015 * df["nps_score"]  # Promoters don't churn
        - 0.01 * np.sqrt(df["tenure_months"])  # Tenure matters
        + 0.002 * df["contract_value"]  # Small companies churn less (proportionally)
        + 0.05 * df["support_tickets"]  # Issues increase churn
    )

    # Heterogeneous treatment effect (CATE)
    # Discount works better for:
    # - Mid-size companies (sweet spot)
    # - Moderate engagement
    # - Not already loyal customers
    treatment_effect = (
        -1.2  # Average treatment effect (reduces churn, logit scale)
        - 0.0001 * (df["company_size"] - 200) ** 2  # Best for mid-size
        + 0.01 * df["usage_score"]  # Less effective for highly engaged users
        + 0.008 * np.abs(df["nps_score"])  # Less effective for extreme NPS (very happy/unhappy)
    )

    # Final churn probability on logit scale
    churn_logit = base_churn_logit + treatment_effect * df["discount_offered"]
    churn_prob = stats.logistic.cdf(churn_logit)

    # Generate binary churn outcome
    churned = (np.random.random(n_samples) < churn_prob).astype(int)

    # Add small label noise
    flip_mask = np.random.random(n_samples) < 0.02  # 2% random flips
    churned = np.where(flip_mask, 1 - churned, churned)
    df["churned"] = churned

    # Store true treatment effect for validation (in practice, unknown)
    df["true_treatment_effect"] = treatment_effect

    # =========================================================
    # SURVIVAL EXTENSION: time_to_churn + churn_event
    # =========================================================
    # We simulate time-to-event with treatment and risk effects.
    # Higher risk -> shorter time; treatment reduces risk -> longer time.

    # Baseline hazard proxy from base churn logit
    # Convert to a positive rate; scale controls typical time horizon.
    # Use a simple mapping: higher logit -> higher hazard -> shorter survival.
    base_churn_logit_clipped = np.clip(base_churn_logit, -10, 10)
    hazard_base = np.exp(base_churn_logit_clipped)
    hazard_base = hazard_base / np.median(hazard_base) # normalize around 1

    # Treatment effect on hazard: discount reduces hazard where it helps
    # Use the same treatment_effect signal but dampened and in hazard space.
    treatment_effect_clipped = np.clip(treatment_effect, -10, 10)
    hazard_multiplier_treat = np.exp(-0.5 * treatment_effect_clipped)  # <1 if treatment_effect < 0

    hazard = hazard_base * np.where(df["discount_offered"] == 1, hazard_multiplier_treat, 1.0)

    # Simulate time-to-churn from Exponential with rate = hazard / scale
    # Choose scale so that typical times are in a reasonable range (e.g., ~365 days).
    time_scale = 365.0
    lam = hazard / time_scale
    raw_time = np.random.exponential(1.0 / lam)

    # Apply administrative censoring at a maximum follow-up time
    max_followup = 730.0  # 2 years
    time_to_churn = np.minimum(raw_time, max_followup)
    churn_event = (raw_time <= max_followup).astype(int)

    df["time_to_churn"] = time_to_churn
    df["churn_event"] = churn_event

    return df


if __name__ == "__main__":
    # Generate data
    print("Generating B2B customer data...")

    df = generate_b2b_customer_data(n_samples=5000)

    # Save original-style dataset
    df.to_csv("b2b_customer_data.csv", index=False)
    print(f"✓ Generated {len(df)} customer records -> b2b_customer_data.csv")

    # Save survival-augmented dataset
    # Note: You'll need b2b_customer_data_survival.csv with time_to_churn and churn_event
    # for the survival CATE script.
    df.to_csv("b2b_customer_data_survival.csv", index=False)
    print("✓ Saved survival dataset with time_to_churn and churn_event -> "
          "b2b_customer_data_survival.csv")

    # Basic statistics
    print(f"\nTreatment rate: {df['discount_offered'].mean():.1%}")
    print(f"Overall churn rate: {df['churned'].mean():.1%}")
    print(f"Churn rate (discount): "
          f"{df[df['discount_offered'] == 1]['churned'].mean():.1%}")
    print(f"Churn rate (no discount): "
          f"{df[df['discount_offered'] == 0]['churned'].mean():.1%}")
    naive_effect = (
        df[df["discount_offered"] == 1]["churned"].mean()
        - df[df["discount_offered"] == 0]["churned"].mean()
    )
    print(f"Naive effect: {naive_effect * 100:.1f} pp")
    print(f"True ATE (logit scale): {df['true_treatment_effect'].mean():.3f}")

    # Survival summary
    print(f"\nMedian time_to_churn/censoring: {df['time_to_churn'].median():.1f} days")
    print(f"Event rate (churn_event=1): {df['churn_event'].mean():.1%}")
    print("Done.")
