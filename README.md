# STATS19 vehicle combinations and casualty multiplicity

Research companion for **Vehicle combinations and multiple fatal or serious casualties in collisions involving two vehicles in Great Britain**.

Authors: Mingming Dong and Ge Chen. This manuscript is being prepared for submission to PLOS ONE; this repository does not indicate acceptance or publication.

## Availability during peer review

This public repository contains aggregate reporting tables, figure source values, figures, and selected analysis source files. It does **not** yet contain the frozen collision-level analysis input or the 50 completed imputation states. The complete computational review package is prepared separately for confidential delivery through the journal. A public GitHub URL alone is not access to that confidential package.

The authors will make the data and author-generated code necessary to reproduce all reported findings publicly available no later than publication. Unrelated drafts, operational logs and author administrative files are outside that release. A versioned archive with a persistent identifier should accompany the publication release. No DOI has been assigned to this review-stage repository.

## What can be verified now

Run from the repository root with Python 3.9 or later; this command needs no third-party packages:

```console
python code/verify_public_results.py
```

The script checks file hashes, cohort totals, the two reported burden percentages, all 186 vehicle contrasts against the full coefficient table, and the rounded primary odds ratios. It checks consistency of the saved reports; it does not refit models, validate the missing-at-random assumption, or establish causal interpretation.

`code/engine.py` and `code/analyze.py` are selected original statistical source files. Running their models requires the confidential review input, saved states and dependencies; the public repository by itself is not a complete model-reproduction environment.

## Files and units

| Location | Contents |
| --- | --- |
| `tables/Table1_source.csv` | Vehicle-combination collision counts and recorded KSI casualty counts |
| `tables/vehicle_contrasts.csv` | 186 vehicle contrasts across 62 specifications |
| `tables/all_regression_coefficients.csv` | 4,832 coefficient rows, including adjustment parameters |
| `tables/per_imputation_model_fit.csv` | 2,904 fit records |
| `tables/Fig1_source.csv`, `tables/Fig2_source.csv`, `tables/FigA_source.csv` | Exact saved figure source values |
| Other `tables/` files | Model inventory, coding labels, sensitivity scope, missingness, diagnostics, STROBE locations and exploratory standardisation |
| `figures/` | Original SVG figures from the manuscript package; Fig A belongs to S1 Appendix |
| `code/` | Selected original model code and a lightweight aggregate-report verifier |
| `MANIFEST.json` | SHA-256 inventory; generated outputs and the manifest itself are excluded |

Primary analysis: 50,735 Great Britain collisions in 2021-2024 with exactly two linked vehicles and at least one police-recorded killed or seriously injured (KSI) casualty. There are 4,119 multiple-KSI collisions (8.12%), accounting for 9,537 of 56,153 recorded KSI casualties (16.98%).

The reference group is two cars/taxis. Primary adjusted odds ratios round to 0.52 for car/taxi-HGV, 0.14 for car/taxi-motorcycle and 0.24 for other known combinations. They describe collision-level multiplicity conditional on recorded KSI, not vehicle safety rankings or individual risks. The known-age subset excludes all 13,446 cycle-involving collisions in the other-known group; its different estimate is not an isolated vehicle-age effect. Poisson results are exploratory conditional additional-KSI mean ratios, not logistic odds ratios. Adjustment-parameter p values are not additional primary hypotheses.

## Data provenance and reproducibility boundaries

Original STATS19 data: UK Department for Transport, [Road safety open data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data). Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/), except where otherwise indicated by the source.

The analysis uses frozen inputs. Current DfT downloads can differ after revisions; source hashes and frozen linked analysis inputs are retained in the confidential review package. Statistical imputations are estimates, not newly observed driver information. The review package reproduces analyses from these frozen linked inputs; the upstream raw acquisition/linkage audit archive is not contained in the supplied materials.

There are 1,243 collision records shared with a separate, unsubmitted motorway-severity manuscript. This work is not an independent replication. AI assistance in code, numerical checking, English drafting and editing is disclosed in the manuscript; the authors retain responsibility for the work.

## Licensing

Author-written source code is offered under `LICENSE` (MIT). Original derived aggregate reporting tables and figures are offered under CC BY 4.0 as described in `DATA_LICENSE.md`; upstream DfT rights and attribution remain applicable. These licenses do not change the access status of files that are not in this public repository.
