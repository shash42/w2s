import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
from metrics import get_jsd, get_kappa_mcqs, get_diffp, populate_preds, precomputed_accs, precomputed_diffs
from visualizations import get_preds_all_modelpairs

def plot_scatter(df, x_col, y_col, groupbydataset=True, hue_col='Dataset', annotate_modelpairs=False, model_short_form=None, title=None, savepath=None):
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette='tab10',
        s=100,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    
    # Add regression lines & R²
    
    r2_dict = {}
    if groupbydataset:
        for label in df[hue_col].unique():
            label_df = df[df[hue_col] == label]
            if len(label_df) < 2:
                continue  # Can't fit a line if less than 2 points
            sns.regplot(data=label_df, x=x_col, y=y_col, scatter=False, ci=None, line_kws={'linestyle': '--'})
            
            X = label_df[x_col].values.reshape(-1, 1)
            Y = label_df[y_col].values
            reg = LinearRegression().fit(X, Y)
            r2 = reg.score(X, Y)
            correlation = 'Positive' if reg.coef_[0] > 0 else 'Negative' if reg.coef_[0] < 0 else 'Zero'
            r2_dict[label] = f"{correlation} (R²={r2:.2f})"
    else:
        sns.regplot(data=df, x=x_col, y=y_col, scatter=False, ci=None, line_kws={'linestyle': '--'})
        
        X = df[x_col].values.reshape(-1, 1)
        Y = df[y_col].values
        reg = LinearRegression().fit(X, Y)
        r2 = reg.score(X, Y)
        correlation = 'Positive' if reg.coef_[0] > 0 else 'Negative' if reg.coef_[0] < 0 else 'Zero'
        r2_dict[hue_col] = f"{correlation} (R²={r2:.2f})"
        print(r2)
    
    # Annotate model pairs if requested
    if annotate_modelpairs and model_short_form is not None:
        for _, row in df.iterrows():
            weak_short = model_short_form.get(row['WeakModel'], row['WeakModel'])
            strong_short = model_short_form.get(row['StrongModel'], row['StrongModel'])
            annotation = f"{weak_short}_{strong_short}"
            plt.text(
                row[x_col],
                row[y_col] + 0.1,
                annotation,
                fontsize=9,
                ha='center',
                va='bottom'
            )
    
    # Update legend
    handles, labels = scatter.get_legend_handles_labels()
    new_labels = [f"{label}: {r2_dict.get(label, '')}" for label in labels]
    plt.legend(
        handles=handles,
        labels=new_labels,
        title=f'{hue_col} (with R²)',
        fontsize=10,
        title_fontsize=13,
        loc='best'
    )
    
    plt.xlabel(x_col, fontsize=14)
    plt.ylabel(y_col, fontsize=14)
    if title is not None:
        plt.title(title, fontsize=16)
    else:
        plt.title(f"{y_col} vs. {x_col}", fontsize=16)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if savepath is None: plt.show()
    else: 
        plt.savefig(savepath)
        print(f"Saved plot to {savepath}")

def get_diffs(folder_name, model_names, datasets, skip_list):
    diffs_test = {}

    for metric_func, metric_name in zip([get_jsd, get_kappa_mcqs, get_diffp, get_diffp, get_diffp], ["JSD", "Kappa", "Prediction Diff%", "Accuracy Gap", "Mag Accuracy Gap"]):
        diffs_test[metric_name] = {}
        for (weak_model, strong_model), preds in preds_dict.items():
            if (weak_model, strong_model) in skip_list:
                continue
            if "Accuracy Gap" in metric_name:
                acc_val = {dset: precomputed_accs(preds, f"{folder_name}/{weak_model}___{strong_model}", dset, model_names, "val") for dset in datasets}
                diffs_test[metric_name][(weak_model, strong_model)] = {dset: acc_val[dset]["strong_base"] - acc_val[dset]["weak_ft"] for dset in datasets}
                if "Mag" in metric_name:
                    diffs_test[metric_name][(weak_model, strong_model)] = {dset: np.abs(v) for dset, v in diffs_test[metric_name][(weak_model, strong_model)].items()}
                continue
            print(weak_model, strong_model)
            diffs_test[metric_name][(weak_model, strong_model)] = {dset : precomputed_diffs(preds, f"{folder_name}/{weak_model}___{strong_model}", dset, "val", metric_func, metric_name, "weak_ft", "strong_base") for dset in datasets}    
    return diffs_test

def populate_df(folder_name, model_names, datasets, skip_list, preds_dict, diffs_test, base_columns):
    data_records = []

    # Build the unified DataFrame
    for (weak_model, strong_model), preds in preds_dict.items():
        if (weak_model, strong_model) in skip_list:
            continue
        
        print(f"Processing Models: {weak_model}, {strong_model}")
        
        # Compute accuracy dict for each dataset
        acc_test = {
            dset: precomputed_accs(
                preds,
                f"{folder_name}/{weak_model}___{strong_model}",
                dset,
                model_names,
                "test"
            )
            for dset in datasets
        }
        
        # Compute metric values for each dataset
        # Assuming diffs_test is structured as: diffs_test[metric_name][(weak_model, strong_model)][dset]
        for dset in datasets:
            # Extract required accuracies
            strong_base_acc = acc_test[dset]["strong_base"]
            weak_ft_acc = acc_test[dset]["weak_ft"]
            w2s_acc = acc_test[dset]["w2s"]
            strong_ft_acc = acc_test[dset]["strong_ft"]
            
            # Compute metrics
            # Make sure these metric functions accept the appropriate parameters.
            # Typically, they might need predictions or something similar.
            # Adjust the calls as needed based on how get_jsd, get_kappa_mcqs, get_diffp are defined.
            jsd_value = diffs_test["JSD"][(weak_model, strong_model)][dset]   # Example usage
            kappa_value = diffs_test["Kappa"][(weak_model, strong_model)][dset]
            diffp_value = diffs_test["Prediction Diff%"][(weak_model, strong_model)][dset]
            
            record = {
                "WeakModel": weak_model,
                "StrongModel": strong_model,
                "Dataset": dset,
                "StrongBaseAcc": strong_base_acc,
                "WeakFtAcc": weak_ft_acc,
                "W2SAcc": w2s_acc,
                "StrongFtAcc": strong_ft_acc,
                "JSD": jsd_value,
                "Kappa": kappa_value,
                "DiffP": diffp_value
            }
            data_records.append(record)

    df_base = pd.DataFrame(data_records, columns=base_columns)

    # Now df_base is your unified DataFrame with all the requested base columns.
    print(df_base.head())
    return df_base

if __name__ == "__main__":
    folder_name = "epochs_3"
    model_names = ["weak_ft", "strong_base", "w2s", "strong_ft"]
    # If you only want a subset, uncomment below:
    # model_names = ["weak_ft", "strong_base", "w2s", "strong_ft"]
    datasplits = ["val", "test"]
    datasets = ["anli-r2", "boolq", "cola", "ethics-utilitarianism", "sciq", "piqa", "sst2", "twitter-sentiment"]
    skip_list = []

    preds_dict = get_preds_all_modelpairs(folder_name, model_names, datasplits=datasplits)

    # Updated loop to handle legend-based R^2 values
    skip_list = [("gemma-2-9b", "Llama-3.1-8B"), ("Llama-3.1-8B", "gemma-2-9b"), ("Llama-3.1-8B", "Llama-3.1-8B"), ("Qwen2.5-0.5B", "OLMo-2-1124-7B")]
    diffs_test = get_diffs(folder_name, model_names, datasets, skip_list)

    # Example model short form dictionary
    model_short_form = {
        "Llama-3.1-8B": "L.1-8",
        "gemma-2-9b": "G-9",
        "phi-2": "P-2.7",
        "gemma-2-2b": "G-2",
        "Qwen2.5-7B": "Q-7",
        "Llama-3.2-1B": "L.2-1",
        "Qwen2.5-0.5B": "Q-0.5",
        "SmolLM-1.7B": "S-1.7",
        # Add more models as needed
    }

    # Define the metrics you want to compute
    metrics = {
        "JSD": get_jsd,
        "Kappa": get_kappa_mcqs,
        "Prediction Diff%": get_diffp
    }

    # The base DataFrame columns you requested
    base_columns = [
        "WeakModel", "StrongModel", "Dataset",
        "StrongBaseAcc", "WeakFtAcc", "W2SAcc", "StrongFtAcc",
        "JSD", "Kappa", "DiffP"
    ]

    df_base = populate_df(folder_name, model_names, datasets, skip_list, preds_dict, diffs_test, base_columns)

    df_base['mean_gain'] = df_base['W2SAcc'] - (df_base['WeakFtAcc'] + df_base['StrongBaseAcc']) / 2
    df_minus_piqa = df_base[df_base['Dataset'] != 'piqa']
    # df_base.plot.scatter(x='DiffP', y='mean_gain', hue='Dataset')
    plot_scatter(
        df_minus_piqa,
        x_col='JSD',
        y_col='mean_gain',
        groupbydataset=False,
        hue_col='Dataset',
        annotate_modelpairs=False,
        model_short_form=model_short_form,
        title="Kappa vs. Mean Gain",
        savepath=f"results/{folder_name}/kappa_vs_mean_gain.png"
    )