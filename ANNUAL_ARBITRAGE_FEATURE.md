# Annual vs. Monthly Arbitrage Feature

## Overview
Added a new AI agent tool called `annual_arbitrage` that calculates potential savings by switching from monthly to annual billing plans.

## How It Works

### Criteria
The tool analyzes subscriptions that meet ALL of the following conditions:
1. **Currently on monthly billing** - Only monthly subscriptions are analyzed
2. **12+ months of payment history** - Calculated from `start_date` to `last_billed_date`
3. **Both pricing options available** - Merchant must have both `monthly_amount` and `annual_amount` in the metadata

### Calculations
For each qualifying subscription:
- **Annual Cost if Monthly**: `monthly_amount × 12`
- **Annual Savings**: `(monthly_amount × 12) - annual_amount`
- **Savings Percentage**: `(annual_savings / annual_cost_if_monthly) × 100`

### Output
The tool returns an InsightCard with:
- **Total potential annual savings** across all qualifying subscriptions
- **Average savings percentage**
- **Number of opportunities found**
- **Detailed breakdown** for each subscription including:
  - Merchant name
  - Plan name
  - Category
  - Months active (payment history length)
  - Monthly amount
  - Annual amount
  - Annual cost if staying monthly
  - Annual savings amount
  - Savings percentage

### Severity Levels
- **High**: Total potential savings ≥ $100
- **Medium**: Total potential savings ≥ $50
- **Low**: Total potential savings < $50

### Recommended Actions
The tool suggests:
1. Switch to annual billing for subscriptions you plan to keep long-term
2. Calculate break-even: if you cancel before 12 months, you may lose money on annual plans
3. Contact merchant support to switch billing frequency—some offer prorated upgrades

## Integration with AI Agent

The tool is automatically invoked when users ask about:
- Saving money (combined with `monthly_commitment`, `duplicates`, `price_mismatch`)
- Annual vs monthly savings specifically
- Optimization opportunities

## Example Query
```bash
venv/bin/python sub_ai_insights_agent.py \
  --subs user_identified_subscriptions.csv \
  --merchants subscription_merchants.csv \
  --asof 2025-11-30 \
  --query "How much money can I save by switching to annual plans?" \
  --out insights_annual.json
```

## Data Requirements

### Subscription CSV
Must include:
- `billing_frequency` (or `billing_frequency_norm`)
- `start_date`
- `last_billed_date`
- `subscription_amount`

### Merchant CSV
Must include:
- `monthly_amount` - Monthly plan price
- `annual_amount` - Annual plan price

## Example Output

```json
{
  "id": "annual_arbitrage",
  "title": "Annual vs. Monthly Savings Opportunities",
  "severity": "high",
  "confidence": 0.85,
  "facts": {
    "subscriptions_with_12plus_months": 5,
    "savings_opportunities_found": 4,
    "total_potential_annual_savings": 156.48,
    "average_savings_pct": 15.2
  },
  "recommended_actions": [
    "Switch to annual billing for subscriptions you plan to keep long-term.",
    "Calculate break-even: if you cancel before 12 months, you may lose money on annual plans.",
    "Contact merchant support to switch billing frequency—some offer prorated upgrades."
  ],
  "supporting_items": [
    {
      "subscription_id": "S001",
      "merchant_name": "Netflix",
      "plan_name_norm": "Standard",
      "category_norm": "Streaming",
      "months_active": 13.5,
      "monthly_amt": 16.49,
      "annual_amt": 164.90,
      "annual_cost_if_monthly": 197.88,
      "annual_savings": 32.98,
      "savings_pct": 16.67
    }
  ]
}
```

## Notes
- The feature only flags subscriptions with **positive savings** (where annual is cheaper than 12 months of monthly)
- Subscriptions without 12 months of history are excluded to ensure users have demonstrated long-term commitment
- Missing pricing data in merchant metadata will result in subscriptions being skipped
