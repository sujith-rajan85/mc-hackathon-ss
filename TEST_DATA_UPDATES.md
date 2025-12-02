# Test Data Updates for Annual Arbitrage Feature

## Summary
Updated both CSV files to include test data with 12+ months of payment history to properly test the annual arbitrage savings feature.

## Changes Made

### user_identified_subscriptions.csv

Updated the following subscriptions to have 12+ months of history (start_date moved to 2024):

| Subscription ID | Merchant | Start Date | Last Billed | Months Active | Monthly Amount |
|-----------------|----------|------------|-------------|---------------|----------------|
| S001 | Netflix | 2024-10-05 | 2025-11-05 | ~13 months | $16.49 |
| S002 | Disney+ | 2024-09-10 | 2025-11-10 | ~14 months | $11.99 |
| S004 | Spotify | 2024-08-07 | 2025-11-07 | ~15 months | $10.99 |
| S005 | Apple Music | 2024-07-15 | 2025-11-15 | ~16 months | $10.99 |
| S006 | iCloud+ | 2024-06-01 | 2025-11-01 | ~17 months | $3.99 |
| S007 | Google One | 2024-05-20 | 2025-11-20 | ~18 months | $2.79 |
| S008 | Dropbox | 2024-04-01 | 2025-11-01 | ~19 months | $15.99 |

### subscription_merchants.csv

Added missing annual pricing:

| Merchant ID | Merchant | Monthly Amount | Annual Amount | Annual Savings |
|-------------|----------|----------------|---------------|----------------|
| 1 | Netflix | $16.49 | $164.90 | $32.98 (16.67%) |

## Test Results

Running the query: **"How much money can I save by switching to annual plans?"**

### Results:
- **Total Potential Savings**: $145.92/year
- **Subscriptions Analyzed**: 7 with 12+ months history
- **Savings Opportunities**: 7 subscriptions
- **Average Savings**: 16.6%

### Detailed Breakdown:

1. **Netflix** (Standard)
   - Monthly: $16.49 × 12 = $197.88
   - Annual: $164.90
   - **Savings: $32.98 (16.67%)**

2. **Dropbox** (Plus)
   - Monthly: $15.99 × 12 = $191.88
   - Annual: $159.99
   - **Savings: $31.89 (16.62%)**

3. **Disney+** (Premium)
   - Monthly: $11.99 × 12 = $143.88
   - Annual: $119.99
   - **Savings: $23.89 (16.60%)**

4. **Spotify** (Individual)
   - Monthly: $10.99 × 12 = $131.88
   - Annual: $109.99
   - **Savings: $21.89 (16.60%)**

5. **Apple Music** (Individual)
   - Monthly: $10.99 × 12 = $131.88
   - Annual: $109.99
   - **Savings: $21.89 (16.60%)**

6. **iCloud+** (200GB)
   - Monthly: $3.99 × 12 = $47.88
   - Annual: $39.99
   - **Savings: $7.89 (16.48%)**

7. **Google One** (100GB)
   - Monthly: $2.79 × 12 = $33.48
   - Annual: $27.99
   - **Savings: $5.49 (16.40%)**

## Verification

The feature correctly:
✅ Identifies subscriptions with 12+ months of payment history
✅ Calculates exact savings amounts and percentages
✅ Only flags subscriptions where annual plans save money
✅ Sorts results by savings amount (highest first)
✅ Provides actionable recommendations
✅ Returns high severity (total savings > $100)

## Command to Test

```bash
venv/bin/python sub_ai_insights_agent.py \
  --subs user_identified_subscriptions.csv \
  --merchants subscription_merchants.csv \
  --asof 2025-11-30 \
  --query "How much money can I save by switching to annual plans?" \
  --out insights_annual_test.json
```
