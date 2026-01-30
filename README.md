# Causal ML Learning Kit: B2B Discount Campaign

## Business Use Case

**Problem**: A B2B SaaS company offers discounts to at-risk customers to reduce churn. But which customers should receive discounts?

**Challenge**: Simply comparing churn rates between discount recipients and non-recipients is misleading due to **selection bias** - the sales team already targets high-risk, high-value customers.

**Solution**: Use causal machine learning to:
1. Estimate true treatment effect (does discount actually work?)
2. Identify which customer segments benefit most (heterogeneous effects)
3. Optimize discount targeting strategy (uplift modeling)

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate causal_ml
```

### 2. Generate Data

```bash
python generate_data.py
```

Creates `b2b_customer_data.csv` with 5,000 synthetic customers including:
- Customer features (company size, contract value, usage, NPS, tenure)
- Treatment assignment (discount offered: yes/no)
- Outcome (churned: yes/no)
- **Realistic confounding** (high-value at-risk customers get discounts)

### 3. Exploratory Analysis

```bash
python 01_eda.py
```

Performs comprehensive EDA:
- Data overview and distributions
- Treatment assignment patterns
- Selection bias detection (comparing treated vs control groups)
- Visualizations saved to `eda_analysis.png`

**Key Finding**: Naive comparison shows discounts appear ineffective or harmful - but this is due to confounding!

### 4. Causal Analysis

```bash
python 02_causal_analysis.py
```

Applies multiple causal methods:
1. **IPW (Inverse Propensity Weighting)**: Adjusts for confounding via propensity scores
2. **S-Learner**: Single model with treatment as feature
3. **T-Learner**: Separate models for treatment/control groups
4. **X-Learner**: Cross-fitted estimator (best for heterogeneous effects)
5. **Causal Forest**: Non-parametric CATE estimation

**Output**: 
- Average Treatment Effect (ATE) - overall impact
- Conditional ATE (CATE) - individual-level predictions
- Customer segmentation by uplift
- Business recommendations

## Key Concepts

### Causal Inference Fundamentals

**Problem**: Correlation ≠ Causation
- Observing Y after T doesn't mean T caused Y
- Confounders affect both treatment assignment and outcome

**Solution**: Estimate counterfactual - what would have happened without treatment?

**Average Treatment Effect (ATE)**:
```
ATE = E[Y(1) - Y(0)]
     = E[Y|T=1] - E[Y|T=0]  (only valid in RCT!)
```

**Conditional ATE (CATE)**:
```
CATE(x) = E[Y(1) - Y(0) | X=x]
```

Individual-level treatment effect given features X.

### Why Causal ML?

Traditional ML predicts: P(Y|X)

Causal ML answers:
- **Effect estimation**: What's the impact of T on Y?
- **Heterogeneity**: Does effect vary by customer?
- **Targeting**: Who benefits most from treatment?

### Methods Overview

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **IPW** | Simple, estimates ATE | Unstable with extreme propensities | Quick ATE estimate |
| **S-Learner** | Simple, leverages all data | Weak if treatment effect is small | Baseline |
| **T-Learner** | Flexible, easy to implement | Inefficient if group sizes differ | Balanced data |
| **X-Learner** | Handles imbalanced data, efficient | More complex | Heterogeneous effects |
| **Causal Forest** | Non-parametric, adaptive | Computationally intensive | Complex interactions |

## Expected Results

**Naive Analysis** (WRONG):
- Discount group churn: ~25%
- Control group churn: ~20%
- Naive effect: +5 pp (appears harmful!)

**Causal Analysis** (CORRECT):
- True ATE: -8 to -12 pp (reduces churn)
- Heterogeneous effects: 
  - High uplift: Mid-size companies, moderate engagement
  - Low uplift: Very small or very large companies, extreme NPS

**Business Impact**:
- Target ~40-50% of customers with highest uplift
- Expected churn reduction: 8-12 pp in targeted segment
- Avoid wasting discounts on negative-uplift customers

## Next Steps

1. **Sensitivity Analysis**: Test robustness to hidden confounders
2. **Confidence Intervals**: Bootstrap or asymptotic inference
3. **Policy Learning**: Optimize treatment assignment rules
4. **A/B Testing**: Validate predictions with controlled experiment
5. **Real Data**: Apply to actual customer data

## Key Packages

- **EconML**: Microsoft's causal ML library (meta-learners, DML, causal forests)
- **DoWhy**: Causal inference with graphical models
- **Scikit-learn**: Base ML models
- **Pandas/NumPy**: Data manipulation

## References

- EconML documentation: https://econml.azurewebsites.net/
- Athey & Imbens (2016): "Recursive Partitioning for Heterogeneous Causal Effects"
- Künzel et al. (2019): "Metalearners for estimating heterogeneous treatment effects"
