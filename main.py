import numpy as np
import yaml
import random

from composite_likelihood import etstimate_theta
from calculate_overlapping import estimate_r
from calculate_weight import estimate_weight
from data import DataSets
from propensityscore import propensityscore
from trim import target_trim, source_trim
from result import Result

# 0.初期設定
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)
split = config["hyperparameters"]["n_split"]
n_trials = config["hyperparameters"]["n_trial"]
est_r = np.empty(split)
est_lamda = np.empty(split)
est_eta = np.empty(split)
N_k = np.empty(split)
results = []
A = config["hyperparameters"]["source_borrow_number"]
for j in range(0, n_trials):
    # 1.データ作成
    random.seed(j)
    np.random.seed(j)

    data = DataSets(config=config)

    data.generate_data()
    target_x = data.target_x
    source_x = data.source_x
    target_y = data.target_y
    source_y = data.source_y
    # 2.傾向スコアの推定
    ppscore_target, ppscore_source = propensityscore(
        target_x=target_x[:, 1:],
        target_y=target_y,
        source_x=source_x[:, 1:],
        source_y=source_y,
    )

    # 3.傾向スコアに基づいてデータを層別化・トリミング
    target_data, split_values = target_trim(
        target_x, target_y.reshape(-1), ppscore_target
    )
    source_data = source_trim(
        source_x, source_y.reshape(-1), ppscore_source, split_values
    )

    # 4.r_j_kの推定
    for i in range(0, split):
        target_condition = target_data["Group"] == i + 1
        source_condition = source_data["Group"] == i + 1
        target_group = target_data[target_condition]
        source_group = source_data[source_condition]
        est_r_k = estimate_r(
            target_group["Propensity_Score"].to_numpy(),
            source_group["Propensity_Score"].to_numpy(),
        )
        N_k[i] = len(source_group)
        est_r[i] = est_r_k[0]
    sum_r_1 = np.sum(est_r)

    # 5.Λ_jの推定
    for i in range(0, split):
        est_eta[i] = estimate_weight(sum_r=sum_r_1, r_k=est_r[i], A=A, N_j_k=N_k[i])
    est_capital_lamda = np.sum(est_eta)

    # 7.λ_jの推定
    for i in range(0, 5):
        est_lamda[i] = estimate_weight(sum_r=sum_r_1, r_k=est_r[i], A=A, N_j_k=N_k[i])

    # 4.層ごとに平均治療効果を推定
    weighted_theta_sum = 0
    total_weight = 0

    for i in range(0, split):
        target_condition = target_data["Group"] == i + 1
        source_condition = source_data["Group"] == i + 1
        target_group = target_data[target_condition]
        source_group = source_data[source_condition]

        if len(target_group) > 0 and len(source_group) > 0:
            average_treatment_effect = np.mean(
                target_group["Outcomes"].to_numpy()
            ) - np.mean(source_group["Outcomes"].to_numpy())
            weighted_theta_sum += average_treatment_effect * est_lamda[i]
            total_weight += est_lamda[i]

    if total_weight > 0:
        est_theta = weighted_theta_sum / total_weight
    else:
        est_theta = np.nan
    results.append({"trial": j + 1, "estimate_value": est_theta})

Result.save_results(results, config)
