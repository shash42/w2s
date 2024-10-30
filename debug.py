from datasets import Dataset
from scipy.spatial.distance import jensenshannon as jsd
import numpy as np

def get_logits(ds):
    return np.array([1-np.array(ds["soft_pred"]), np.array(ds["soft_pred"])]).T

def get_jsd(ds1, ds2):
    return np.mean(jsd(get_logits(ds1), get_logits(ds2)))

def get_kappa_mcqs(m1, m2, n_options=4):
    same, total, m1_corr, m2_corr, samewrong = 0, 0, 0, 0, 0
    for s1, s2 in zip(m1, m2):
        total += 1
        if s1["pred"] == s2["pred"]:
            same += 1
        if s1["pred"] == s1["labels"]:
            m1_corr += 1
        if s2["pred"] == s2["labels"]:
            m2_corr += 1
        if s1["pred"] == s2["pred"] and s1["pred"] != s1["labels"]:
            samewrong += 1
    cobs = same/total
    p1, p2 = (m1_corr/total), (m2_corr/total)
    expsamewrong = (1 - p1) * (1 - p2) * 1/(n_options - 1)
    cexp = p1 * p2 + expsamewrong
    kappa = (cobs - cexp) / (1 - cexp)
    return kappa

def get_acc(ds):
    labels = np.array([d["labels"] for d in ds])
    preds = np.array([d["soft_pred"] for d in ds])
    return np.mean(labels == (preds > 0.5))*100

def add_preds(ds):
    # Initialize predictions as a list
    preds = []
    for d in ds:
        # Add prediction to the list based on `soft_pred` value
        preds.append(d["soft_pred"] > 0.5)
    # Add the updated predictions list as a new column to the dataset
    ds = ds.add_column("pred", preds)
    return ds

def acc_buckets(dsref1, dsref2, dsmain):
    buckets = {"cc" : {"correct": 0, "total": 0}, "cw": {"correct": 0, "total": 0}, "wc": {"correct": 0, "total": 0}, "ww": {"correct": 0, "total": 0}}
    for d1, d2, dsm in zip(dsref1, dsref2, dsmain):
        if d1["labels"] == d1["pred"]:
            if d2["labels"] == d2["pred"]: # cc
                buckets["cc"]["total"] += 1
                if dsm["labels"] == dsm["pred"]:
                    buckets["cc"]["correct"] += 1
            else: # cw
                buckets["cw"]["total"] += 1
                if dsm["labels"] == dsm["pred"]:
                    buckets["cw"]["correct"] += 1

        else:
            if d2["labels"] == d2["pred"]: # wc
                buckets["wc"]["total"] += 1
                if dsm["labels"] == dsm["pred"]:
                    buckets["wc"]["correct"] += 1
            else:
                buckets["ww"]["total"] += 1
                if dsm["labels"] == dsm["pred"]:
                    buckets["ww"]["correct"] += 1
    
    print(buckets)
    return buckets

dsref1 = Dataset.from_file("results/shared/sciq/weak/predictions/test/data-00000-of-00001.arrow")
print(f"Loaded {len(dsref1)} examples for dsref1, accuracy: {get_acc(dsref1):.2f}%")
dsref2 = Dataset.from_file("results/shared/sciq/strong/predictions/test/data-00000-of-00001.arrow")
print(f"Loaded {len(dsref2)} examples for dsref2, accuracy: {get_acc(dsref2):.2f}%")
dsmain = Dataset.from_file("results/test1/sciq/w2s/predictions/test/data-00000-of-00001.arrow")
print(f"Loaded {len(dsmain)} examples for dsmain, accuracy: {get_acc(dsmain):.2f}%")

dsref1, dsref2, dsmain = add_preds(dsref1), add_preds(dsref2), add_preds(dsmain)
buckets = acc_buckets(dsref1, dsref2, dsmain)
#pretty print bucket accuracies and totals
for cat in buckets:
    print(f"Category {cat}: Accuracy - {buckets[cat]['correct']/buckets[cat]['total']*100:.2f}% Total - {buckets[cat]['total']}")

print(f"JSD between weak_ft and strong_ft: {get_jsd(dsref1, dsref2):.4f}")
print(f"JSD between weak_ft and w2s: {get_jsd(dsref1, dsmain):.4f}")
print(f"JSD between strong_ft and w2s: {get_jsd(dsref2, dsmain):.4f}")

print(f"Kappa between weak_ft and strong_ft: {get_kappa_mcqs(dsref1, dsref2, n_options=2):.4f}")
print(f"Kappa between weak_ft and w2s: {get_kappa_mcqs(dsref1, dsmain, n_options=2):.4f}")
print(f"Kappa between strong_ft and w2s: {get_kappa_mcqs(dsref2, dsmain, n_options=2):.4f}")

