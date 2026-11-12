"""Post-M1 review analysis (exploratory, read-only).

Re-reads the saved M1 experiment scores and model and quantifies the weaknesses discussed in
docs/architecture-review-post-m1.md. The test split has already been looked at, so NOTHING here
may be used to select a model or to claim performance; it only informs what to fix next.

Run from the backend directory after `prepare` and `train` (paths are the M1 defaults and the
experiment version evaluated in the review):

    cd backend && uv run python ../docs/analysis/post_m1_review.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from factorymind.anomaly.evaluation import AlarmPolicy, episodes, evaluate
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from factorymind.anomaly.service import load_bundle
from factorymind.data import metropt3 as m

RUN = Path(r"D:\FactoryMind\artifacts\experiments\metropt3-anomaly\20261110-6413a9f4f3")
DATA = Path(r"D:\FactoryMind\data\processed")
MODELS = Path(r"D:\FactoryMind\artifacts\models")
res = json.load(open(RUN / "results.json"))
W = pd.Timedelta("10min")
H24 = pd.Timedelta(hours=24)
reports = m.FAILURE_REPORTS
DETS = ["lps_rule", "robust_z", "isolation_forest", "pca_spe"]
KW = {"early_horizon": pd.Timedelta(hours=6), "post_tolerance": H24}

sc = {s: pd.read_parquet(RUN / f"scores_{s}.parquet") for s in ("train", "calibration", "validation", "test")}
pooled = pd.concat([sc["validation"], sc["test"]])
idx = pd.DatetimeIndex(pooled.index)
near = np.zeros(len(idx), bool)
for r in reports:
    near |= (idx >= pd.Timestamp(r.start) - H24) & (idx <= pd.Timestamp(r.end) + H24)
neg = ~near
print(f"pooled windows {len(idx)}, far-from-report normal windows {int(neg.sum())}")


def in_report(index: pd.DatetimeIndex, r: m.FailureReport) -> np.ndarray:
    return np.asarray((index <= pd.Timestamp(r.end)) & (index + W > pd.Timestamp(r.start)))


def auroc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    ranks = pd.Series(np.concatenate([pos_scores, neg_scores])).rank().to_numpy()
    n1, n0 = len(pos_scores), len(neg_scores)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


print("\n[A] Threshold-free separability (AUROC in-report vs far-normal) and window recall at chosen threshold")
for d in DETS:
    thr = res["detectors"][d]["threshold"]
    s = pooled[d].to_numpy()
    row = [
        f"{r.report_id}: AUC={auroc(s[in_report(idx, r)], s[neg]):.3f} "
        f"rec={np.mean(s[in_report(idx, r)] > thr):.2f} n={int(in_report(idx, r).sum())}"
        for r in reports
    ]
    print(f"{d:<17}", " | ".join(row))

print("\n[B] Score dynamic range inside reports")
for d in ("isolation_forest", "pca_spe"):
    thr = res["detectors"][d]["threshold"]
    far = pooled[d].to_numpy()[neg]
    print(f"{d} thr={thr:.3f}; far-normal p50={np.median(far):.3f} p99={np.quantile(far, .99):.3f} max={far.max():.3f}")
    for r in reports:
        s = pooled[d].to_numpy()[in_report(idx, r)]
        print(f"   {r.report_id}: p10={np.quantile(s, .1):.3f} p50={np.median(s):.3f} p90={np.quantile(s, .9):.3f} max={s.max():.3f}")

print("\n[C] Fixed chosen policies applied to pooled validation+test (4 reports, post-hoc, descriptive only)")
for d in DETS:
    det = res["detectors"][d]
    r_ = evaluate(idx, pooled[d].to_numpy(), W, AlarmPolicy(det["threshold"], det["persistence"]), reports, **KW)
    print(f"{d:<17} chosen policy: {r_.events_detected}/4  FA/day={r_.false_alarms_per_day:.3f} (eps {r_.false_alarm_episodes})  AP={r_.average_precision:.3f}")
cal = sc["calibration"]
print("   common policy q=0.995 from calibration scores:")
for d in DETS:
    thr = float(np.quantile(cal[d], 0.995))
    for k in (3, 6):
        r_ = evaluate(idx, pooled[d].to_numpy(), W, AlarmPolicy(thr, k), reports, **KW)
        print(f"   {d:<17} k={k}: {r_.events_detected}/4  FA/day={r_.false_alarms_per_day:.3f}  delays={[e.delay_minutes for e in r_.events]}")

print("\n[D] False-alarm anatomy (chosen policies, pooled)")
feats = window_features(m.load_processed(DATA), WindowSpec())
for d in ("isolation_forest", "pca_spe"):
    det = res["detectors"][d]
    eps = episodes(idx, pooled[d].to_numpy(), W, AlarmPolicy(det["threshold"], det["persistence"]))
    rows = []
    for e in eps:
        if any(
            e.start <= pd.Timestamp(r.end) + H24 and e.end >= pd.Timestamp(r.start) - KW["early_horizon"]
            for r in reports
        ):
            continue
        w = feats.loc[(feats.index >= e.start) & (feats.index < e.end)]
        rows.append((str(e.start)[:16], round((e.end - e.start).total_seconds() / 3600, 1), round(float(w["load_duty"].mean()), 2)))
    print(d, "false-alarm episodes (start, hours, mean load duty):", rows)

print("\n[E] Drift: monthly score level of far-normal windows vs calibration")
for d in ("isolation_forest", "pca_spe"):
    s = pooled[d][neg]
    mo = s.groupby(pd.DatetimeIndex(s.index).to_period("M")).agg(["median", lambda x: x.quantile(0.95), "count"])
    mo.columns = ["median", "p95", "n"]
    print(d, "calibration p50/p95:", round(cal[d].median(), 3), round(cal[d].quantile(0.95), 3))
    print(mo.round(3).to_string())

print("\n[F] Evidence redundancy and observability of physics features")
bundle, _card = load_bundle(MODELS, "metropt3-anomaly")
sensors = sorted(m.SENSORS, key=len, reverse=True)


def family(feature: str) -> str:
    return next((s for s in sensors if feature.startswith(s + "_")), feature)


inrep = np.zeros(len(feats), bool)
for r in reports:
    inrep |= in_report(pd.DatetimeIndex(feats.index), r)
fv = feats[feats.valid & inrep]
alarmed = fv[bundle.detector.score(fv[list(FEATURE_NAMES)]) > bundle.policy.threshold]
z = bundle.profile.z(alarmed[list(FEATURE_NAMES)]).abs()
top1: Counter[str] = Counter()
distinct = []
for _, row in z.iterrows():
    top = list(row.sort_values(ascending=False).index[:5])
    top1.update(top[:1])
    distinct.append(len({family(f) for f in top}))
print(f"alarmed in-report windows: {len(alarmed)}; mean distinct sensors in top-5 evidence: {np.mean(distinct):.2f}")
print("most frequent #1 evidence feature:", top1.most_common(4))
normal = feats[feats.valid & ~inrep]
print("unloaded_tp3_slope observable share: normal", round(normal["unloaded_tp3_slope"].notna().mean(), 3),
      "| in-report", round(fv["unloaded_tp3_slope"].notna().mean(), 3))
print("load_duty median: normal", round(normal["load_duty"].median(), 3), "| in-report", round(fv["load_duty"].median(), 3))
print("share with load_duty>0.5: in-report", round((fv["load_duty"] > 0.5).mean(), 3),
      "| normal", round((normal["load_duty"] > 0.5).mean(), 3))

print("\n[G] Physics baseline: compressor load duty alone")
f = feats[feats.valid]
cal_duty = f.loc["2020-03-17":"2020-03-31", "load_duty"]
pool = f.loc["2020-04-01":"2020-09-01"]
pidx = pd.DatetimeIndex(pool.index)
duty = pool["load_duty"].to_numpy()
pneg = np.ones(len(pidx), bool)
for r in reports:
    pneg &= ~((pidx >= pd.Timestamp(r.start) - H24) & (pidx <= pd.Timestamp(r.end) + H24))
print("calibration load_duty q50/q99/q995/max:", [round(float(cal_duty.quantile(q)), 3) for q in (.5, .99, .995, 1)])
print("AUROC per event:", [round(auroc(duty[in_report(pidx, r)], duty[pneg]), 3) for r in reports])
for q in (0.99, 0.995):
    thr = float(cal_duty.quantile(q))
    for k in (3, 6):
        r_ = evaluate(pidx, duty, W, AlarmPolicy(thr, k), reports, **KW)
        print(f"calibration-quantile rule q={q} thr={thr:.3f} k={k}: {r_.events_detected}/4 FA/day={r_.false_alarms_per_day:.3f}")
for thr in (0.3, 0.4, 0.5):
    r_ = evaluate(pidx, duty, W, AlarmPolicy(thr, 6), reports, **KW)
    print(f"fixed threshold {thr} k=6 (chosen post hoc!): {r_.events_detected}/4 FA/day={r_.false_alarms_per_day:.3f} delays={[e.delay_minutes for e in r_.events]}")
