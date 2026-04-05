import json
import numpy as np
import pandas as pd
from pathlib import Path

fname = Path(__file__).parent / "CCBM"/ "outputs"/ "factor_ablation_results"/"factor_ablation_results.json"

with open(fname) as f:
    data = json.load(f)

def calcValue(data_item):
    results = []
    for curr_item in data_item:
        lpips_score = curr_item["lpips_distance_perturb"] / curr_item["lpips_distance_orig"]
        mse_score = curr_item["mse_distance_perturb"] / curr_item["mse_distance_orig"]
        results.append([lpips_score, mse_score])
    return np.array(results)

mse_data = {}
lpips_data = {}

for s_factor in data.keys():
    mse_data[s_factor] = {}
    lpips_data[s_factor] = {}
    for guidance in data[s_factor].keys():
        values = calcValue(data[s_factor][guidance])
        lpips_data[s_factor][guidance] = values[:, 0].mean()
        mse_data[s_factor][guidance] = values[:, 1].mean()

mse_data = pd.DataFrame(mse_data)
lpips_data = pd.DataFrame(lpips_data)