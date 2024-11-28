import os
from datasets import Dataset
from regex import F
from scipy.spatial.distance import jensenshannon as jsd
import numpy as np
from w2s.roc_auc import roc_auc
import torch
import json
import pyarrow as pa



def get_logits(ds):
    return np.array([1-np.array(ds["soft_pred"]), np.array(ds["soft_pred"])]).T

def get_brier(ds):
    labels = np.array([d["labels"] for d in ds])
    preds = np.array([d["soft_pred"] for d in ds])
    return np.mean((labels - preds)**2)

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

def get_diffp(ds1, ds2):
    #s1[pred] != s2['pred']
    diffp = 0
    for s1, s2 in zip(ds1, ds2):
        if s1["pred"] != s2["pred"]:
            diffp += 1
    return diffp/len(ds1)

def get_acc(ds):
    labels = np.array([d["labels"] for d in ds])
    preds = np.array([d["soft_pred"] for d in ds])
    return np.mean(labels == (preds > 0.5))*100

def compute_mean_preds(preds, dname, split):
    preds[dname]["mean"] = {}
    preds[dname]["mean"][split] = preds[dname]["weak_ft"][split]
    preds[dname]["mean"][split] = preds[dname]["mean"][split].remove_columns(["soft_pred", "pred"])
    avg_soft_preds = [(x + y)/2 for (x, y) in zip(preds[dname]["weak_ft"][split]["soft_pred"], preds[dname]["strong_base"][split]["soft_pred"])]
    preds[dname]["mean"][split] = preds[dname]["mean"][split].add_column("soft_pred", avg_soft_preds)
    avg_preds = [x > 0.5 for x in avg_soft_preds]
    preds[dname]["mean"][split] = preds[dname]["mean"][split].add_column("pred", avg_preds)
    return preds[dname]["mean"]
    
def compute_diff_ceil(preds, dname, split):
    preds[dname]["diff_ceil"] = {}
    preds[dname]["diff_ceil"][split] = preds[dname]["weak_ft"][split]
    oracle_preds = list(preds[dname]["weak_ft"][split]["pred"])
    oracle_soft_preds = list(preds[dname]["weak_ft"][split]["soft_pred"])
    for idx, x in enumerate(preds[dname]["strong_base"][split]["pred"]):
        if x == preds[dname]["weak_ft"][split]["labels"][idx]:
            oracle_preds[idx] = x
            oracle_soft_preds[idx] = preds[dname]["strong_base"][split]["soft_pred"][idx]
    
    preds[dname]["diff_ceil"][split] = preds[dname]["diff_ceil"][split].remove_columns(["soft_pred", "pred"])
    preds[dname]["diff_ceil"][split] = preds[dname]["diff_ceil"][split].add_column("soft_pred", oracle_soft_preds)
    preds[dname]["diff_ceil"][split] = preds[dname]["diff_ceil"][split].add_column("pred", oracle_preds)
    return preds[dname]["diff_ceil"]

def populate_preds(preds, datasets, model_names, datasplits, folder_name, weak_model, strong_model, verbose=False):
    if not verbose:
        from datasets.utils.logging import set_verbosity_error
        set_verbosity_error()
    for dname in datasets:
        if verbose: print(f"Dataset: {dname}")
        preds[dname] = {}
        for mname in model_names:
            if verbose: print(f"Model: {mname}")
            mfilename = f"{weak_model}___{strong_model}"
            preds[dname][mname] = {}
            mtype = mname
            if mname.endswith("ft"):
                mtype = mname[:-3]
            
            if "weak" in mname:
                mfilename = weak_model
            elif "strong" in mname:
                mfilename = strong_model                                    
            for split in datasplits:
                preds_path = f"results/{folder_name}/{mfilename}/{dname}/{mtype}/predictions/{split}/data-00000-of-00001.arrow"
                
                if mname == "mean" and not os.path.exists(preds_path):
                    compute_mean_preds(preds, dname, split)
                    continue
                elif mname == "diff_ceil":
                    compute_diff_ceil(preds, dname, split)
                    continue
                try:
                    preds[dname][mname][split] = add_preds(Dataset.from_file(preds_path))
                    if verbose: print(f"Loaded {len(preds[dname][mname][split])} examples for {split} split")
                except:
                    print(f"Error loading {dname} predictions for ({weak_model}, {strong_model})")
    return preds

def precomputed_accs(preds, foldername, dname, mnames, split):
    path = f"results/{foldername}/{dname}/{split}/accs.json"
    #make path if doesnt exist
    os.makedirs(f"results/{foldername}/{dname}/{split}", exist_ok=True)
    accs = {}
    if not os.path.exists(path):
        for mname in mnames:
            accs[mname] = get_acc(preds[dname][mname][split])
        with open(path, "w") as f:
            json.dump(accs, f)
    else:
        with open(path, "r") as f:
            accs = json.load(f)
        for mname in mnames:
            if mname not in accs:
                accs[mname] = get_acc(preds[dname][mname][split])
        with open(path, "w") as f:
            json.dump(accs, f)
    return accs

def precomputed_diffs(preds, foldername, dname, split, diff_func, diff_func_name):
    path = f"results/{foldername}/{dname}/{split}/diffs.json"
    diffs = {}
    os.makedirs(f"results/{foldername}/{dname}/{split}", exist_ok=True)
    if not os.path.exists(path):
        diffs[diff_func_name] = diff_func(preds[dname]["weak_ft"][split], preds[dname]["strong_base"][split])
        with open(path, "w") as f:
            json.dump(diffs, f)
    else:
        with open(path, "r") as f:
            diffs = json.load(f)
        if diff_func_name not in diffs:
            diffs[diff_func_name] = diff_func(preds[dname]["weak_ft"][split], preds[dname]["strong_base"][split])
            with open(path, "w") as f:
                json.dump(diffs, f) 
    return diffs[diff_func_name]

def get_auc(ds):
    labels = torch.from_numpy(np.array([d["labels"] for d in ds])).float()
    preds = torch.from_numpy(np.array([d["soft_pred"] for d in ds])).float()
    return roc_auc(labels, preds)

def add_preds(ds):
    # Initialize predictions as a list
    ds = ds.map(lambda x: {'pred': x['soft_pred'] > 0.5})
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

def pretty_print(dsref1, dsref2, dsmain):
    dsref1, dsref2, dsmain = add_preds(dsref1), add_preds(dsref2), add_preds(dsmain)
    buckets = acc_buckets(dsref1, dsref2, dsmain)
    for cat in buckets:
        print(f"Category {cat}: Accuracy - {buckets[cat]['correct']/buckets[cat]['total']*100:.2f}%, Total - {buckets[cat]['total']}")

    print(f"JSD between {m1name} and {m2name}: {get_jsd(dsref1, dsref2):.4f}")
    print(f"JSD between {m1name} and {m3name}: {get_jsd(dsref1, dsmain):.4f}")
    print(f"JSD between {m2name} and {m3name}: {get_jsd(dsref2, dsmain):.4f}")

    print(f"Kappa between {m1name} and {m2name}: {get_kappa_mcqs(dsref1, dsref2, n_options=2):.4f}")
    print(f"Kappa between {m1name} and {m3name}: {get_kappa_mcqs(dsref1, dsmain, n_options=2):.4f}")
    print(f"Kappa between {m2name} and {m3name}: {get_kappa_mcqs(dsref2, dsmain, n_options=2):.4f}")

if __name__ == '__main__':
    datasets = ['anli-r2', 'boolq', 'cola', 'ethics-utilitarianism', 'hellaswag', 'sciq', 'piqa', 'sst2', 'twitter-sentiment']
    shared_folder_name = 'shared3'
    test_folder_name = 'test3'
    for dset in datasets:
        m1name, m2name, m3name, m4name = "weak", "strong_base", "w2s", "strong"
        print("Dataset:", dset)
        dsref1 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m1name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref1)} examples for {m1name}, accuracy: {get_acc(dsref1):.2f}%, auc: {get_auc(dsref1):.2f}")
        dsref2 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m2name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref2)} examples for {m2name}, accuracy: {get_acc(dsref2):.2f}%, auc: {get_auc(dsref2):.2f}")
        dsmain = Dataset.from_file(f"results/{test_folder_name}/{dset}/{m3name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsmain)} examples for {m3name}, accuracy: {get_acc(dsmain):.2f}%, auc: {get_auc(dsmain):.2f}")
        dsref3 = Dataset.from_file(f"results/{shared_folder_name}/{dset}/{m4name}/predictions/test/data-00000-of-00001.arrow")
        print(f"Loaded {len(dsref1)} examples for {m4name}, accuracy: {get_acc(dsref3):.2f}%, auc: {get_auc(dsref3):.2f}")
        
        pretty_print(dsref1, dsref2, dsmain)
    

