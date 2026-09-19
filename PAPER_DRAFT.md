# Cost-Aware Drift Detection and Automated Retraining for Production ML

## III. Methodology

We study the problem of maintaining a deployed classifier's accuracy under
concept drift while minimizing the operational cost of retraining. Our system
couples an online classifier with a pluggable drift detector: the detector
consumes a binary correctness signal after each prediction and, upon flagging
drift, triggers a retrain on a bounded buffer of the most recent labeled
samples, after which the retrained model hot-swaps in as the serving version.
We instrument every inference and retraining event with a simulated cloud
cost, calibrated to representative 2026 list pricing for serverless inference
(~\$0.20 per 1M invocations, matching AWS Lambda) and a small managed training
job (~\$0.05 per retrain). This lets us report the accuracy/detection
trade-off each detector makes in the same units a deployment decision is
actually made in: dollars.

### A. Drift detectors compared

We compare four published, complementary drift-detection algorithms:

- **DDM** (Drift Detection Method) [Gama et al., 2004] tracks the online
  error rate and its standard deviation, signaling drift when the error rate
  exceeds its historical minimum by a fixed number of standard deviations.
- **EDDM** (Early Drift Detection Method) [Baena-García et al., 2006]
  tracks the distance between consecutive misclassifications rather than the
  raw error rate, making it more sensitive to gradual drift at the cost of
  more false positives.
- **ADWIN** (Adaptive Windowing) [Bifet & Gavaldà, 2007] maintains a variable-
  length window of recent observations, using an exponential-histogram
  compression scheme and a Hoeffding-bound cut-point test to detect a
  statistically significant change in the window's mean.
- **KSWIN** (Kolmogorov-Smirnov Windowing) [Raab et al., 2020] applies a
  two-sample KS test between a reference and a recent sub-window of a
  sliding window, requiring no distributional assumption on the correctness
  signal.

All four were implemented from scratch against their original publications
(numpy/scipy only), rather than via a drift-detection library, so that the
cost-instrumentation and hot-swap retraining logic could be integrated
directly into the online serving loop.

### B. Synthetic drift streams

Real-world drift benchmarks such as Elec2 lack a labeled ground-truth drift
point, which makes *detection delay* — the number of samples between the
true onset of drift and the detector's alarm — impossible to measure
rigorously. We therefore evaluate on three synthetic streams standard in the
drift-detection literature, each with known drift points:

| Stream | Type | Drift points | Description |
|---|---|---|---|
| SEA (abrupt) | abrupt | 3 | Threshold on sum of two features changes discretely |
| SINE (abrupt) | abrupt | 3 | Decision boundary (`f2 > sin(f1)`) reverses |
| Rotating Hyperplane (gradual) | gradual | 2 | Hyperplane weights rotate smoothly over a 1500-sample window |

Each stream has 20,000 samples, a 500-sample warm-start for the initial
model fit, a 500-sample retraining buffer, and a 200-sample cooldown between
retrains to prevent thrashing on a single drift event. The base classifier is
a linear `SGDClassifier` (log-loss), retrained from scratch on the buffer
each time a drift is confirmed.

## IV. Results

Table I reports, per (stream, detector) pair: drifts detected out of the
known ground truth, mean detection delay in samples, false alarms (flags not
matched to a true drift point), overall post-warm-up accuracy, and total
simulated cost.

**Table I: Detector comparison across synthetic drift streams**

| Stream | Detector | Detected | Mean Delay | False Alarms | Accuracy | Cost (USD) |
|---|---|---|---|---|---|---|
| SEA (abrupt) | DDM | 3/3 | 2574.0 | 0 | 0.786 | 0.154 |
| SEA (abrupt) | EDDM | 3/3 | 4000.3 | 13 | 0.676 | 0.804 |
| SEA (abrupt) | ADWIN | 3/3 | 3465.7 | 2 | 0.745 | 0.254 |
| SEA (abrupt) | KSWIN | 1/3 | 2496.0 | 0 | 0.733 | 0.054 |
| SINE (abrupt) | DDM | 2/3 | 264.5 | 0 | 0.575 | 0.104 |
| SINE (abrupt) | EDDM | 3/3 | 477.3 | 7 | 0.633 | 0.504 |
| SINE (abrupt) | ADWIN | 3/3 | 718.3 | 0 | 0.637 | 0.154 |
| SINE (abrupt) | KSWIN | 3/3 | 67.7 | 0 | 0.437 | 0.154 |
| Hyperplane (gradual) | DDM | 2/2 | 477.5 | 1 | 0.730 | 0.154 |
| Hyperplane (gradual) | EDDM | 2/2 | 90.5 | 22 | 0.817 | 1.204 |
| Hyperplane (gradual) | ADWIN | 2/2 | 747.0 | 2 | 0.785 | 0.204 |
| Hyperplane (gradual) | KSWIN | 1/2 | 3577.0 | 1 | 0.587 | 0.104 |

Aggregating across the three streams (8 total drift events):

| Detector | Drifts detected | Total false alarms | Mean accuracy | Mean cost/run (USD) |
|---|---|---|---|---|
| DDM | 7/8 | 1 | 0.697 | 0.137 |
| EDDM | 8/8 | 42 | 0.709 | 0.837 |
| **ADWIN** | **8/8** | **4** | **0.722** | **0.204** |
| KSWIN | 5/8 | 1 | 0.586 | 0.104 |

## V. Discussion

The results expose a clear, quantifiable trade-off along three axes —
detection completeness, false-alarm rate, and cost — rather than a single
"best" detector:

- **EDDM** achieves full detection (8/8) but at the price of 42 false
  alarms in total, an order of magnitude more than ADWIN, driving its mean
  retraining cost to \$0.837/run — over 4x ADWIN's. Its sensitivity to the
  distance between errors makes it react fastest under gradual drift
  (90.5-sample delay on the hyperplane stream) but this same sensitivity to
  noise inflates false alarms on the abrupt streams.
- **DDM** is the cheapest reliable option, with only a single false alarm
  across all 12 runs, but it fails to detect one of the three abrupt drifts
  in the SINE stream entirely (2/3), which would leave a production model
  silently stale.
- **KSWIN** is the cheapest detector overall but the least reliable,
  missing 3 of 8 total drift events (including entirely missing 2 of 3 in
  SEA and 1 of 2 in the hyperplane stream) — its non-parametric two-sample
  test appears under-powered against these correctness-signal
  distributions at the tested window size.
- **ADWIN** is the only detector to achieve full detection (8/8) across
  every stream while keeping false alarms low (4 total, vs. EDDM's 42),
  at roughly one-quarter of EDDM's average cost. This makes it the best
  accuracy-cost operating point among the four for a deployment that
  cannot tolerate missed drift but is retraining-cost-sensitive.

These findings argue that detector choice for a production self-healing
pipeline should not default to whichever algorithm has the lowest published
detection delay — DDM and KSWIN both post competitive delays but at the cost
of missed drifts — but should instead be selected against the specific
cost-of-missed-drift vs. cost-of-false-retrain trade-off of the deployment.

## VI. Limitations and Threats to Validity

- Streams are synthetic; while standard in the drift-detection literature
  for enabling ground-truth detection-delay measurement, results on a
  real-world stream (e.g., a production traffic log with a known policy
  change date) may differ, particularly for KSWIN's sensitivity to the
  correctness-signal's window statistics.
- Cost figures are simulated from representative list pricing, not measured
  cloud billing; Section VII reports early results from the deployed
  version of this pipeline on Render's free tier as a step toward closing
  this gap with real infrastructure numbers.
- The retraining buffer size (500 samples) and cooldown (200 samples) were
  fixed across all experiments; both parameters likely interact with
  detector choice and warrant a sensitivity analysis in future work.

## References (to merge into refs.bib)

- J. Gama, P. Medas, G. Castillo, and P. Rodrigues, "Learning with Drift
  Detection," in *Advances in Artificial Intelligence – SBIA 2004*.
- M. Baena-García, J. del Campo-Ávila, R. Fidalgo, A. Bifet, R. Gavaldà,
  and R. Morales-Bueno, "Early Drift Detection Method," in *ECML PKDD 2006
  Workshop on Knowledge Discovery from Data Streams*.
- A. Bifet and R. Gavaldà, "Learning from Time-Changing Data with Adaptive
  Windowing," in *Proc. SIAM Int. Conf. on Data Mining (SDM)*, 2007.
- C. Raab, M. Heusinger, and F.-M. Schleif, "Reactive Soft Prototype
  Computing for Concept Drift Streams," *Neurocomputing*, 2020. (KSWIN)
