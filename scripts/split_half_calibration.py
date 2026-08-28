"""
Threshold-circularity fix: split-half calibration analysis.

Splits the existing 150-case melanoma validation cohort (which already has
per-case pleu, ecs_score, confidence, is_correct saved) into a calibration
half and a held-out half. Recalibrates the ECS decision threshold and the
PLEU rejection fraction on the calibration half ONLY, then reports AUROC,
catch rate, and false-alarm rate on the untouched held-out half.

This directly answers the "thresholds were calibrated and evaluated on the
same set" criticism for these two thresholds. It does NOT require any new
model inference — everything needed is already in the saved CSV.

Note: this script covers ECS's tau and PLEU's rejection fraction (rho),
since both are directly computable from the existing melanoma CSV. tau_u
(the patch-level uncertainty threshold) and tau_ood (OOD threshold) were
calibrated on different, non-melanoma-specific pools (all validation patches;
synthetic OOD images respectively) and are not covered by this script --
if you want to fix those too, that needs a separate script and a small
amount of new inference.

Run from your project root:
    python -m scripts.split_half_calibration
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ==================== CONFIG ====================
MELANOMA_CSV = "results/ecs_melanoma_results_with_stats_v2.csv"
OUTPUT_TXT = "results/split_half_calibration_results.txt"
RANDOM_SEED = 42  # matches your paper's existing seed convention
N_BOOTSTRAP = 1000  # for CIs on the held-out half, matching your paper's existing bootstrap convention


def youdens_j_threshold(scores, y_correct, higher_score_means_correct=True):
    """
    Find the threshold on `scores` that maximizes Youden's J (sensitivity +
    specificity - 1) for distinguishing correct (1) vs misclassified (0) cases.

    Returns: (best_threshold, best_J)
    """
    candidate_thresholds = np.unique(scores)
    best_j = -np.inf
    best_thresh = None

    for t in candidate_thresholds:
        if higher_score_means_correct:
            pred_correct = scores >= t
        else:
            pred_correct = scores <= t

        tp = np.sum((pred_correct == 1) & (y_correct == 1))
        fn = np.sum((pred_correct == 0) & (y_correct == 1))
        tn = np.sum((pred_correct == 0) & (y_correct == 0))
        fp = np.sum((pred_correct == 1) & (y_correct == 0))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sensitivity + specificity - 1

        if j > best_j:
            best_j = j
            best_thresh = t

    return best_thresh, best_j


def evaluate_at_threshold(scores, y_correct, threshold, higher_score_means_correct=True):
    """
    Given a fixed threshold (chosen elsewhere), compute catch rate (recall on
    misclassified cases) and false-alarm rate (1 - specificity) on this data.
    'Catch' = correctly flagging a misclassified case as low-trust.
    """
    if higher_score_means_correct:
        flagged_as_untrustworthy = scores < threshold
    else:
        flagged_as_untrustworthy = scores > threshold

    is_misclassified = (y_correct == 0)
    is_correct = (y_correct == 1)

    n_misclassified = is_misclassified.sum()
    n_correct = is_correct.sum()

    caught = np.sum(flagged_as_untrustworthy & is_misclassified)
    false_alarms = np.sum(flagged_as_untrustworthy & is_correct)

    catch_rate = caught / n_misclassified if n_misclassified > 0 else np.nan
    false_alarm_rate = false_alarms / n_correct if n_correct > 0 else np.nan

    return catch_rate, false_alarm_rate, n_misclassified, n_correct


def bootstrap_ci_auroc(scores, y_correct, n_boot=1000, seed=42):
    """95% CI for AUROC via bootstrap resampling."""
    rng = np.random.RandomState(seed)
    n = len(scores)
    aurocs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y_correct[idx]
        s_boot = scores[idx]
        if len(np.unique(y_boot)) < 2:
            continue  # skip degenerate resamples with only one class
        try:
            aurocs.append(roc_auc_score(y_boot, s_boot))
        except ValueError:
            continue
    if len(aurocs) == 0:
        return np.nan, np.nan
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)


def main():
    if not os.path.isfile(MELANOMA_CSV):
        raise FileNotFoundError(f"MELANOMA_CSV not found: {MELANOMA_CSV}")

    df = pd.read_csv(MELANOMA_CSV)
    required_cols = ['pleu', 'ecs_score', 'confidence', 'is_correct']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    n_total = len(df)
    print(f"Loaded {n_total} melanoma validation cases.")
    print(f"Overall correct: {df['is_correct'].sum()}, misclassified: {(1 - df['is_correct']).sum()}")

    # ---------------- Split into calibration / held-out halves ----------------
    rng = np.random.RandomState(RANDOM_SEED)
    shuffled_idx = rng.permutation(n_total)
    half = n_total // 2
    calib_idx = shuffled_idx[:half]
    heldout_idx = shuffled_idx[half:]

    df_calib = df.iloc[calib_idx].reset_index(drop=True)
    df_heldout = df.iloc[heldout_idx].reset_index(drop=True)

    print(f"\nCalibration half: n={len(df_calib)} "
          f"(correct={df_calib['is_correct'].sum()}, misclassified={(1-df_calib['is_correct']).sum()})")
    print(f"Held-out half:    n={len(df_heldout)} "
          f"(correct={df_heldout['is_correct'].sum()}, misclassified={(1-df_heldout['is_correct']).sum()})")

    results_lines = []
    results_lines.append(f"n_total: {n_total}")
    results_lines.append(f"n_calibration: {len(df_calib)}")
    results_lines.append(f"n_heldout: {len(df_heldout)}")

    # ==================== ECS THRESHOLD ====================
    print(f"\n{'='*60}")
    print("ECS THRESHOLD — recalibrated on calibration half only")
    print(f"{'='*60}")

    ecs_calib_scores = df_calib['ecs_score'].values
    ecs_calib_y = df_calib['is_correct'].values
    ecs_thresh, ecs_j = youdens_j_threshold(ecs_calib_scores, ecs_calib_y, higher_score_means_correct=True)
    print(f"Recalibrated ECS threshold (Youden's J on calibration half): tau = {ecs_thresh:.3f} (J = {ecs_j:.3f})")
    print(f"  (Original paper value, calibrated on full set: tau = 0.95)")

    ecs_heldout_scores = df_heldout['ecs_score'].values
    ecs_heldout_y = df_heldout['is_correct'].values
    catch, false_alarm, n_mis, n_corr = evaluate_at_threshold(
        ecs_heldout_scores, ecs_heldout_y, ecs_thresh, higher_score_means_correct=True
    )
    ecs_heldout_auroc = roc_auc_score(ecs_heldout_y, ecs_heldout_scores)
    ecs_ci_low, ecs_ci_high = bootstrap_ci_auroc(ecs_heldout_scores, ecs_heldout_y, N_BOOTSTRAP, RANDOM_SEED)

    print(f"\nHeld-out evaluation (n={len(df_heldout)}, {n_mis} misclassified, {n_corr} correct):")
    print(f"  Catch rate: {catch*100:.1f}%")
    print(f"  False-alarm rate: {false_alarm*100:.1f}%")
    print(f"  AUROC: {ecs_heldout_auroc:.3f} [95% CI: {ecs_ci_low:.3f}-{ecs_ci_high:.3f}]")
    print(f"  (Original paper value, evaluated on full set: AUROC 0.531 [95% CI: 0.48-0.58], "
          f"catch rate 70.7%, false-alarm 58.2%)")

    results_lines.append(f"\n--- ECS ---")
    results_lines.append(f"recalibrated_tau: {ecs_thresh:.3f}")
    results_lines.append(f"heldout_catch_rate: {catch:.3f}")
    results_lines.append(f"heldout_false_alarm_rate: {false_alarm:.3f}")
    results_lines.append(f"heldout_auroc: {ecs_heldout_auroc:.3f}")
    results_lines.append(f"heldout_auroc_ci: [{ecs_ci_low:.3f}, {ecs_ci_high:.3f}]")

    # ==================== PLEU REJECTION FRACTION ====================
    print(f"\n{'='*60}")
    print("PLEU REJECTION FRACTION — recalibrated on calibration half only")
    print(f"{'='*60}")

    pleu_calib_scores = df_calib['pleu'].values
    pleu_calib_y = df_calib['is_correct'].values
    # Higher PLEU = more uncertain = LESS likely correct, so higher_score_means_correct=False
    pleu_thresh, pleu_j = youdens_j_threshold(pleu_calib_scores, pleu_calib_y, higher_score_means_correct=False)
    print(f"Recalibrated PLEU rejection threshold (Youden's J on calibration half): rho = {pleu_thresh:.3f} (J = {pleu_j:.3f})")
    print(f"  (Original paper value, calibrated on full set: rho = 0.4)")

    pleu_heldout_scores = df_heldout['pleu'].values
    pleu_heldout_y = df_heldout['is_correct'].values
    catch_p, false_alarm_p, n_mis_p, n_corr_p = evaluate_at_threshold(
        pleu_heldout_scores, pleu_heldout_y, pleu_thresh, higher_score_means_correct=False
    )
    # AUROC: higher PLEU -> more likely misclassified, so use -pleu as the "correctness" score
    pleu_heldout_auroc = roc_auc_score(pleu_heldout_y, -pleu_heldout_scores)
    pleu_ci_low, pleu_ci_high = bootstrap_ci_auroc(-pleu_heldout_scores, pleu_heldout_y, N_BOOTSTRAP, RANDOM_SEED)

    print(f"\nHeld-out evaluation (n={len(df_heldout)}, {n_mis_p} misclassified, {n_corr_p} correct):")
    print(f"  Catch rate: {catch_p*100:.1f}%")
    print(f"  False-alarm rate: {false_alarm_p*100:.1f}%")
    print(f"  AUROC: {pleu_heldout_auroc:.3f} [95% CI: {pleu_ci_low:.3f}-{pleu_ci_high:.3f}]")
    print(f"  (Original paper value, evaluated on full set: AUROC 0.523)")

    results_lines.append(f"\n--- PLEU ---")
    results_lines.append(f"recalibrated_rho: {pleu_thresh:.3f}")
    results_lines.append(f"heldout_catch_rate: {catch_p:.3f}")
    results_lines.append(f"heldout_false_alarm_rate: {false_alarm_p:.3f}")
    results_lines.append(f"heldout_auroc: {pleu_heldout_auroc:.3f}")
    results_lines.append(f"heldout_auroc_ci: [{pleu_ci_low:.3f}, {pleu_ci_high:.3f}]")

    # ==================== SAVE ====================
    with open(OUTPUT_TXT, 'w') as f:
        f.write('\n'.join(results_lines))
    print(f"\n{'='*60}")
    print(f"Saved full results to: {OUTPUT_TXT}")
    print(f"{'='*60}")
    print("\nCompare the held-out numbers above against your original full-set numbers.")
    print("If they're similar (within the CI ranges), that's evidence the original")
    print("thresholds were NOT badly overfit to the evaluation set. If they diverge")
    print("substantially, report the held-out numbers as the more honest estimate.")


if __name__ == "__main__":
    main()