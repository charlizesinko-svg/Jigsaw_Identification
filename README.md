# Jigsaw Identification: Re-identification via Dataset Linkage

Code for the empirical proof-of-concept in **"You Cannot Haunt What You Cannot Find: Jigsaw Identification and the Illusion of Anonymized Data"** by Charlize Sinko.

This analysis demonstrates how joining two individually anonymized public datasets increases record uniqueness, reducing effective k-anonymity guarantees.

---

## Overview

The analysis joins two publicly available datasets across shared quasi-identifiers (age range, occupation, and geographic unit) and measures the change in record uniqueness (k=1 fraction) before and after the join.

**Key result:** Uniqueness rate increased from 48.5% to 70.3% (+21.8 percentage points) after a single auxiliary join — using only public data.

---

## Datasets

Both datasets are freely available and used under their respective public use agreements. No sensitive or private data is used.

**1. ACS PUMS (U.S. Census Bureau)**
- American Community Survey Public Use Microdata Sample, 1-year estimate (2023)
- Download: https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/
- Quasi-identifiers used: age, sex, occupation (SOC code), PUMA, income range

**2. Public Salary Survey (Kaggle)**
- Employment and salary survey dataset
- Source: https://www.kaggle.com/code/ahmedashraf299/data-science-salaries
- Quasi-identifiers used: age range, job title, geographic region, salary range

---

## Requirements
Install dependencies:

```bash
pip install pandas
```

---

## Usage

1. Download both datasets (see links above) and place them in a `data/` folder.
2. Run the analysis script:

```bash
python jigsaw_analysis.py
```

The script will output:
- Baseline uniqueness rate on PUMS alone
- Uniqueness rate after the auxiliary salary survey join
- A summary table matching Table 1 in the paper

---

## Methodology

The analysis proceeds in three steps:

1. **Baseline k-anonymity** — Compute k-anonymity on the PUMS dataset using the quasi-identifier set `{age, sex, occupation, PUMA}` and record the fraction of unique records (k=1).
2. **Auxiliary join** — Perform a quasi-identifier join between PUMS and the salary survey on shared fields: age range, occupation/job title, and geographic unit.
3. **Post-join k-anonymity** — Recompute uniqueness on the augmented dataset and measure the change.

> Note: Records in PUMS that did not match any wage bracket in the salary survey were excluded from the joined dataset, which accounts for the reduction in total records (235,216 → 192,370).

---

## Results Summary

| Condition | Total Records | Unique Records (k=1) | Uniqueness Rate |
|---|---|---|---|
| PUMS Alone | 235,216 | 114,076 | 48.5% |
| After Auxiliary Join | 192,370 | 133,920 | 70.3% |
| Change | -44,846 | +19,844 | +21.8 pp |

---

## Important Note

This analysis does **not** re-identify any actual individuals. It demonstrates only that the structural conditions for re-identification (record uniqueness) are created by the join operation itself. The purpose is to illustrate a policy-relevant vulnerability in static anonymization standards, not to harm anyone's privacy.

---

## Citation

If you use this code or methodology, please cite:

> Sinko, C. (2025). *You Cannot Haunt What You Cannot Find: Jigsaw Identification and the Illusion of Anonymized Data.*

---

## License

This code is released for academic and research purposes.
