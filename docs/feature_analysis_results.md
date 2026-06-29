# Feature Analysis Results

**Script:** `src/scripts/feature_analysis.py`  
**Dataset:** LendingClub Accepted Loans 2007–2018 Q4  
**Sample size:** 100,000 rows, 151 columns

---

## Dataset Overview

| Property | Value |
|---|---|
| Rows analyzed | 100,000 |
| Total columns | 151 |
| Numeric columns | 115 |
| Categorical columns | 36 |

---

## 1. Correlation Analysis

The target column `loan_status` is categorical (e.g. `"Fully Paid"`, `"Charged Off"`) and was not present as a numeric type, so Pearson correlation against the target could not be computed directly.

**Action required:** Encode `loan_status` as a binary integer (e.g. `0 = Fully Paid`, `1 = Charged Off/Default`) before re-running correlation analysis.

---

## 2. Missing Value Analysis

73 out of 151 features have at least one missing value.

### Completely Missing (100%)

These features have no data at all in the sample and should be dropped:

| Feature | Missing % |
|---|---|
| `member_id` | 100.0 |
| `sec_app_num_rev_accts` | 100.0 |
| `sec_app_open_act_il` | 100.0 |
| `sec_app_inq_last_6mths` | 100.0 |
| `sec_app_open_acc` | 100.0 |
| `sec_app_mort_acc` | 100.0 |
| `sec_app_mths_since_last_major_derog` | 100.0 |
| `sec_app_collections_12_mths_ex_med` | 100.0 |
| `sec_app_chargeoff_within_12_mths` | 100.0 |
| `sec_app_fico_range_low` | 100.0 |
| `sec_app_earliest_cr_line` | 100.0 |
| `sec_app_revol_util` | 100.0 |
| `sec_app_fico_range_high` | 100.0 |
| `revol_bal_joint` | 100.0 |
| `desc` | 99.99 |

> All `sec_app_*` and joint application fields are 100% missing, indicating the sample contains virtually no joint or secondary applicant loans.

---

## 3. Variance Analysis

### Zero-Variance Columns (constant values — no predictive power)

| Feature | Note |
|---|---|
| `policy_code` | Always the same value |
| `deferral_term` | Always the same value |
| `hardship_length` | Always the same value |

These three columns carry no information and must be dropped.

### Highest Variance Features

| Feature | Variance |
|---|---|
| `id` | 5.94 × 10¹² |
| `tot_hi_cred_lim` | 3.19 × 10¹⁰ |
| `tot_cur_bal` | 2.48 × 10¹⁰ |
| `annual_inc` | 7.97 × 10⁹ |
| `annual_inc_joint` | 2.79 × 10⁹ |
| `total_bal_ex_mort` | 2.44 × 10⁹ |
| `total_il_high_credit_limit` | 2.04 × 10⁹ |
| `total_bal_il` | 1.86 × 10⁹ |
| `total_rev_hi_lim` | 1.20 × 10⁹ |
| `revol_bal` | 5.62 × 10⁸ |

> Note: `id` has the highest variance but is an identifier — it must be dropped. High variance in balance/income fields is expected; they are valuable but will need scaling.

---

## 4. Categorical Feature Analysis

36 categorical features were found. Key highlights from the first 5:

| Feature | Unique Values | Notes |
|---|---|---|
| `term` | 2 | `36 months` (68%), `60 months` (32%) — low cardinality, easy to encode |
| `grade` | 7 | A–G, well-distributed; strong credit risk signal |
| `sub_grade` | 35 | Finer breakdown of grade; likely redundant with `grade` |
| `emp_title` | 37,529 | Near-unique; too high cardinality for direct encoding — drop or group |
| `emp_length` | 11 | Ordinal categories (`< 1 year` to `10+ years`) — encode as ordinal integer |

---

## 5. Skewness Analysis

46 numeric features have `|skewness| > 2` and should be log- or sqrt-transformed before modeling.

### Most Skewed Features

| Feature | Skewness |
|---|---|
| `delinq_amnt` | 78.1 |
| `tax_liens` | 54.1 |
| `annual_inc` | 51.8 |
| `num_tl_120dpd_2m` | 41.7 |
| `tot_coll_amt` | 27.6 |
| `pub_rec` | 23.3 |
| `total_rec_late_fee` | 22.3 |
| `num_tl_30dpd` | 19.0 |
| `chargeoff_within_12_mths` | 16.2 |
| `acc_now_delinq` | 15.5 |

> Heavy right-skew is typical for financial data (income, delinquency amounts). Apply `log1p` transformation (handles zeros) before feeding these into linear or distance-based models.

---

## 6. Multicollinearity Analysis

21 highly correlated pairs (`|r| > 0.9`) were found among 26 unique features.

### Correlated Pairs

| Feature A | Feature B | Correlation |
|---|---|---|
| `loan_amnt` | `funded_amnt` | 1.000 |
| `loan_amnt` | `funded_amnt_inv` | 1.000 |
| `loan_amnt` | `installment` | 0.944 |
| `funded_amnt` | `funded_amnt_inv` | 1.000 |
| `funded_amnt` | `installment` | 0.944 |
| `funded_amnt_inv` | `installment` | 0.944 |
| `fico_range_low` | `fico_range_high` | 1.000 |
| `open_acc` | `num_sats` | 0.999 |
| `out_prncp` | `out_prncp_inv` | 1.000 |
| `total_pymnt` | `total_pymnt_inv` | 1.000 |

### Recommended Resolutions

| Keep | Drop |
|---|---|
| `loan_amnt` | `funded_amnt`, `funded_amnt_inv` |
| `installment` | (derived from loan_amnt × rate — consider keeping both or dropping one) |
| `fico_range_low` | `fico_range_high` (or average them into `fico_score`) |
| `open_acc` | `num_sats` |
| `out_prncp` | `out_prncp_inv` |
| `total_pymnt` | `total_pymnt_inv` |

---

## 7. Feature Recommendations Summary

### Drop (60 features)

| Category | Features |
|---|---|
| Zero variance | `policy_code`, `deferral_term`, `hardship_length` |
| 100% missing | `member_id`, all `sec_app_*` (13 features), `revol_bal_joint`, `desc` |
| >50% missing | `dti_joint`, `verification_status_joint`, `annual_inc_joint`, `orig_projected_additional_accrued_interest`, `payment_plan_start_date`, all `hardship_*` fields, all `settlement_*` fields, `next_pymnt_d`, `mths_since_last_record`, `il_util`, `mths_since_rcnt_il`, `open_rv_24m`, `inq_last_12m`, `max_bal_bc`, `total_cu_tl`, `total_bal_il`, `open_il_24m`, `open_act_il`, `open_acc_6m`, `inq_fi`, `open_il_12m`, `open_rv_12m`, `all_util`, `mths_since_recent_bc_dlq`, `mths_since_last_major_derog`, `mths_since_recent_revol_delinq` |

### Transform — Apply `log1p` (46 features)

Key examples: `annual_inc`, `delinq_amnt`, `tax_liens`, `tot_coll_amt`, `pub_rec`, `revol_bal`, `total_rev_hi_lim`, `tot_hi_cred_lim`, `total_bal_ex_mort`, `recoveries`, `collection_recovery_fee`, `last_pymnt_amnt`, `total_rec_int`, `out_prncp`, `out_prncp_inv`.

> Note: `id` appears in this list but should be dropped outright, not transformed.

### Investigate for Multicollinearity (26 features)

`loan_amnt`, `funded_amnt`, `funded_amnt_inv`, `installment`, `fico_range_low`, `fico_range_high`, `open_acc`, `num_sats`, `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv` — and 14 others listed in section 6.

---

## Next Steps

1. **Encode target:** Binarize `loan_status` → `0 / 1` and re-run correlation analysis.
2. **Drop** the 60 recommended features.
3. **Encode categoricals:** Ordinal-encode `emp_length`, one-hot encode `term`, `grade`; consider target-encoding `sub_grade`; drop `emp_title` or group into top-N buckets.
4. **Transform** the 46 skewed features with `log1p`.
5. **Resolve multicollinearity** by dropping redundant columns per the table above.
6. **Scale** remaining numeric features (StandardScaler or RobustScaler, given outliers).
7. Implement the stub ETL functions in `lending_club_etl.py` (`drop_columns`, `parse_dates`, `handle_missing_values`) using the findings above.
