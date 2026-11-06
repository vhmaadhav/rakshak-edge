# URL three-seed controls; fixed teacher/KD unmeasurable

**The fixed ByT5 teacher stopped on nonfinite loss; no selected teacher, eligibility or KD targets exist.**
No stronger-teacher or distillation gain is claimed. Preserve the fixed arm as UNMEASURABLE:
no precision/batch/architecture/learning-rate switch. The six original E0 controls remain useful
for the requested three-seed report, without promoting a new model on test outcomes.

E0 cal-only artifacts: url_controls_cal-e0-20261102v1/tree
`ebcc1bb75c6e9668a789928946c7247e66b560305d5e62e38598fe7d675f715b`.
Manifest `88b20cd71febccdc1ed81d8c2a9ec0f1b984b9eed9f44c389f357b5bfcc4ac7d`.
Teacher p1v1/tree `84d35409ac4ac48138a2a38673bd14efd7682b40808d64a9b80762d3f7e5e64d`
error112.8s: `FloatingPointError: nonfinite FP32 teacher loss; stop fixed arm`.
No completed epoch logged; partial optimizer updates and numerical root cause are UNVERIFIED.

## Hypothesis, gates and preserved failures

Pretrained byte context was hypothesized to make a stronger URL-only teacher, enabling train-only KD.
Apache-2.0 google/byt5-small@68377bdc18a2ffec8a0533fef03b1c513a4dd49d was pinned/card checked.
Teacher recipe, cal-only eligibility then exact-fit-ID-only KD, scientific test gains/retention gates
were predeclared in URL_TEACHER_PLAN.md before calibration feedback. Both teacher/KD performance
gates remain UNMEASURABLE; numerical failure is not evidence that a different teacher could not work.
No companion teacher run can rescue the missing primary Phresh arm; none was launched.

Controls use exact frozen fit/cal IDs, original charcnn-wide/200bytes/side features, seeds1/2/3,
8epochs/patience3/calAP, AdamW.002/wd.0001/b512/AMP. Seed set BEFORE model construction;
the inherited URL job set it inside training after construction. Old historical results are not a
matched comparator. No URL expert silently substituted into frozen fusionv1; no new fused FPR claim.

Original inference eval e0evalv1/tree4fdfe6994123 completed allsix test caches, then failed478.4s
on missing `url_temporal` file. All6cal parity checks and thresholds froze before any test scores.
Its exact threshold manifest SHA:
`54e4fa8f9f1d6128ac31f6515b698d768fa8facf98931b00cb3acfd64bf91509`.
First cached recovery e0statsv1/tree3fec836a5fef failed94.9s on a manifest adapter KeyError before
aggregate metrics. Mapping fixed/regression added, scores/thresholds unchanged. Its one fetch
completed file downloads but error-text detection matched a filename; receipt verified manually, no refetch.
Artifact renamed to avoid that false positive. No dataset/split changed and no inference repeated.

Final cache-only e0stats2v1/tree
`17576395e1c2fc76e32d4970c15ca5c180844da3527a40542680e07c6d34b7c2`
completed435.9s, one event/fetch. Exact six score hashes, ordered original fit/cal/test IDs, parent
checkpoint/cal hashes and threshold reconstruction verified before aggregation. No model execution.
**url_temporal is UNMEASURABLE because its file is absent**, not reconfirmed single-class.
Do not invent negatives/splits or quote a previous temporal NA as newly validated.

## Official PhreshPhish protocol, all published rates

URL-only. All975 original subsets:586/296/59/29/5 by base rate. Each subset metric is computed
first, then averaged by rate and over the three fixed seeds. This is not pooled AP or the weighted
calibration proxy. All1000 registered-domain bootstrap draws (seed13) were valid at every rate.
Draw multiplicities are shared across models/subsets; no initialization-seed resampling.
CI uncertainty is conditional on these3fixed seeds and the approximate registered-domain groups.
Ties/zero-weight score knots match original sklearn subset point metrics within1e-12.
Historical benchmark feedback motivated this arm: **in-sample intervention selection**, no untouched
benchmark/SOTA or Indian validation claim. PaperGTE uses URL+HTML; no direct competitor ranking.

| Base rate | Subsets | Three-seed mean AP [domain95%CI] | P@R90 [domain95%CI] | Valid draws AP / P@R90 |
|---|---:|---|---|---|
| 0.05% | 586 | 0.443457 [0.323937, 0.620641] | 0.073388 [0.052072, 0.183454] | 1000 / 1000 |
| 0.1% | 296 | 0.558907 [0.428609, 0.705899] | 0.132927 [0.076425, 0.275599] | 1000 / 1000 |
| 0.5% | 59 | 0.784396 [0.692895, 0.868465] | 0.383532 [0.233850, 0.616371] | 1000 / 1000 |
| 1% | 29 | 0.848300 [0.780365, 0.909560] | 0.549905 [0.366032, 0.762092] | 1000 / 1000 |
| 5% | 5 | 0.937183 [0.907551, 0.963933] | 0.859681 [0.746046, 0.943847] | 1000 / 1000 |

Per-seed values and domain CIs (same original subsets):

| Rate | Seed | AP [CI] | P@R90 [CI] |
|---|---|---|---|
| 0.05% | s1 | 0.503797 [0.372117, 0.664819] | 0.090497 [0.065410, 0.234176] |
| 0.05% | s2 | 0.481218 [0.335983, 0.667807] | 0.067091 [0.046987, 0.168581] |
| 0.05% | s3 | 0.345354 [0.251557, 0.529330] | 0.062575 [0.042689, 0.152455] |
| 0.1% | s1 | 0.621149 [0.489532, 0.752634] | 0.160929 [0.091266, 0.338910] |
| 0.1% | s2 | 0.588803 [0.437739, 0.747821] | 0.121016 [0.069957, 0.255437] |
| 0.1% | s3 | 0.466769 [0.346426, 0.638866] | 0.116838 [0.067718, 0.234179] |
| 0.5% | s1 | 0.820429 [0.744159, 0.890206] | 0.431161 [0.260356, 0.687393] |
| 0.5% | s2 | 0.792989 [0.695021, 0.878599] | 0.358319 [0.216342, 0.591217] |
| 0.5% | s3 | 0.739770 [0.629161, 0.839016] | 0.361116 [0.222611, 0.581360] |
| 1% | s1 | 0.871936 [0.816657, 0.921941] | 0.593693 [0.397800, 0.817748] |
| 1% | s2 | 0.851786 [0.780125, 0.913617] | 0.524229 [0.341923, 0.741649] |
| 1% | s3 | 0.821177 [0.738450, 0.892944] | 0.531795 [0.354539, 0.735676] |
| 5% | s1 | 0.943668 [0.917539, 0.967099] | 0.880916 [0.773267, 0.959158] |
| 5% | s2 | 0.936892 [0.905557, 0.963987] | 0.849266 [0.724252, 0.938633] |
| 5% | s3 | 0.930989 [0.897497, 0.960261] | 0.848862 [0.738820, 0.937013] |

Full benchmark/test (not a substitute for rare-rate subsets):

| Population | Seed / fixed-seed mean | AP [CI] | P@R90 [CI] |
|---|---|---|---|
| full_benchmark | s1 | 0.981149 [0.971247, 0.989495] | 0.978061 [0.954874, 0.993046] |
| full_benchmark | s2 | 0.979681 [0.968992, 0.988686] | 0.971644 [0.941970, 0.989194] |
| full_benchmark | s3 | 0.979267 [0.968481, 0.988449] | 0.971643 [0.945378, 0.989252] |
| full_benchmark | mean | 0.980033 [0.969728, 0.988852] | 0.973783 [0.947625, 0.990411] |
| full_test | s1 | 0.991836 [0.987954, 0.995070] | 0.991011 [0.983993, 0.996479] |
| full_test | s2 | 0.991997 [0.988304, 0.995169] | 0.991096 [0.983956, 0.995876] |
| full_test | s3 | 0.991987 [0.988332, 0.995091] | 0.990273 [0.982938, 0.995507] |
| full_test | mean | 0.991940 [0.988223, 0.995085] | 0.990793 [0.983627, 0.995945] |

## Frozen calibration thresholds and classification endpoints

Every threshold below was frozen before inference, with checkpoint/source/git/kernel/input hashes
in evaluation_manifest.json. Calibration logits reconstructed within unchanged1e-5 full-cal tolerance.
Fixed-cal operating-point rates here are points; url_domain maxF1/AP CIs follow. Do not label
test-oracle recall@FPR intervals as intervals for fixed-cal thresholds.

| Dataset | Seed | Full-cal max logit error | MaxF1 threshold | Test F1 | Test FPR / recall | Cal1% threshold | Test FPR / recall |
|---|---|---|---|---|---|---|---|
| phreshphish | 1 | 0 | 0.477742 | 0.957141 | 0.019516 / 0.939089 | 0.743394 | 0.009040 / 0.911341 |
| phreshphish | 2 | 0 | 0.598456 | 0.958460 | 0.019910 / 0.942005 | 0.870193 | 0.009106 / 0.913906 |
| phreshphish | 3 | 0 | 0.535719 | 0.958294 | 0.022365 / 0.944375 | 0.885164 | 0.007133 / 0.897357 |
| url_domain | 1 | 0 | 0.012486 | 0.999216 | 0.003478 / 0.999564 | 0.003253 | 0.008428 / 0.999652 |
| url_domain | 2 | 0 | 0.001620 | 0.999281 | 0.002542 / 0.999390 | 0.000602 | 0.004281 / 0.999608 |
| url_domain | 3 | 0 | 0.002962 | 0.999564 | 0.001472 / 0.999608 | 0.000539 | 0.004950 / 0.999695 |

url_domain test30,432 rows (22,957 positives), all3seeds:

| Seed | MaxF1 F1 [campaign95%CI] | AP [campaign95%CI] | CI method / clusters |
|---|---|---|---|
| 1 | 0.999216 [0.998953, 0.999449] | 0.999974 [0.999945, 0.999993] | campaign_cluster / 23941 |
| 2 | 0.999281 [0.999004, 0.999520] | 0.999982 [0.999963, 0.999996] | campaign_cluster / 23941 |
| 3 | 0.999564 [0.999341, 0.999750] | 0.999987 [0.999970, 0.999997] | campaign_cluster / 23941 |

## Provenance, validation and decision

Final metricsSHA `0093e20c39a8d372f391d1ffaec05be89950f56bd00563f5bbc1141853d8f5ee`.
Recovery manifestSHA `152096a4847cd97e96402f7ad02f81eef704aeb72d3b1ab9bd38d356f9f343a1`.
All original input-file/split hashes, six checkpoints/cal-logit hashes/IDs and six test-cache hashes
are bound by the parent/recovery manifests. No local real URL scoring/training; local tests use
synthetic fixtures only. Original frozen splits unchanged, no paid service used.

Two-stage focused validation passed, plus synthetic weighted-AP/explicit repeated-domain-row,
checkpoint/tamper, cache/order and manifest-adapter regressions; detailed failures/runs in STATUS.md.
One fullKaggle suite858passed0skipped,157warnings125.18s; ruffclean. Private archive
maadhavvh/rakshak-url-e0-20261102 version1, ownermaadhavvh/isPrivate=true verified,
17,083,574bytes; archive manifestSHA
`ced9776473225eba4d48ef68d574d2f017f50b4bfeb1ed01cb66a1f5f3013234`. Initial processing403 retained, no re-upload.
Private archive allowlist15files: six checkpoints, sourcezip and aggregate/provenance/failureJSON;
no raw URLs or row-score parquets. HF publish pending. No deployment approval.

Keep the three-seed control measurement, stop the fixed teacher/KD arm as UNMEASURABLE.
The rare-base-rate gap remains open. Future teacher/precision designs need a new predeclaration;
these test outcomes cannot choose a rescue recipe. No stronger-teacher, KD win or SOTA claim.
