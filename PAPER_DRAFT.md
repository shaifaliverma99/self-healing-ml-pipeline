# Cost-Aware Drift Detection and Automated Retraining for Production ML

**Superseded.** This file was an early single-run draft. It has been replaced by
the full, statistically validated paper generated from real experiment data:

- **Paper:** `paper/Self_Healing_ML_Pipeline_IEEE.docx` (generate with `python paper/build_paper.py`)
- **Source data:** `results/detector_comparison.csv` (single-run), `results/statistical_validation.csv`
  and `results/statistical_validation_raw.csv` (10-seed validation), `results/significance_tests.csv`
  (paired Wilcoxon tests), `results/ablation_study.csv` (detector-triggered vs. periodic retraining),
  `results/timing.csv` (measured per-detector wall-clock cost)

The early draft's headline claim ("ADWIN catches every drift") was based on a
single run and does not hold under multi-seed statistical testing -- see the
current paper for the corrected, evidence-scoped finding: ADWIN's advantage
over EDDM is a cost advantage at statistically equal accuracy, and its
advantage over DDM/KSWIN is an accuracy advantage that is not always free.
