import numpy as np
import yaml
import random

from composite_likelihood import etstimate_theta
from calculate_overlapping import estimate_r
from calculate_weight import estimate_weight
from data import DataSets
from propensityscore import propensityscore
from trim import target_trim,source_trim

#0.初期設定
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["hyperparam"]["n_split"]
est_theta=np.zeros(config["hyperparam"]["n_trial"])
true_values=np.ones_like(est_theta)
A=config["datasettings"]["source_borrow_number"]
for j in range(0,config["hyperparam"]["n_trial"]):
    #1.データ作成
    random.seed(j)
    np.random.seed(j)

    data=DataSets()   

    est_r=np.empty(split)
    est_lamda=np.empty(split)
    est_eta=np.empty(split)
    N_k=np.empty(split)

    #2.傾向スコアの推定
    ppscore_target,ppscore_source = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_x[:,1:],source_y=data.training_historical_y)

    #3.傾向スコアに基づいてデータを層別化・トリミング
    target_data,split_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),ppscore_target)
    source_data=source_trim(data.training_historical_x,data.training_historical_y.reshape(-1),ppscore_source,split_values)

    #4.r_j_kの推定
    for i in range(0,split):
        target_condition = target_data['Group'] == i+1
        source_condition = source_data['Group'] == i+1
        target_group=target_data[target_condition]
        source_group=source_data[source_condition]
        est_r_k=estimate_r(target_group['Propensity_Score'].to_numpy(),source_group['Propensity_Score'].to_numpy())
        N_k[i]=len(source_group)
        est_r[i]=est_r_k[0]
    sum_r_1=np.sum(est_r)

    #5.Λ_jの推定
    for i in range(0,split):
        est_eta[i]=estimate_weight(sum_r=sum_r_1,r_k=est_r[i],A=A,N_j_k=N_k[i])
    est_capital_lamda=np.sum(est_eta) 

     #7.λ_jの推定
    for i in range(0,5):
        est_lamda[i]=estimate_weight(sum_r=sum_r_1,r_k=est_r[i],A=A,N_j_k=N_k[i])

    #4.層ごとに平均治療効果を推定
    for i in range(0,split):
        target_condition = target_data['Group'] == i+1
        source_condition = source_data['Group'] == i+1  
        target_group=target_data[target_condition]
        source_group=source_data[source_condition]
        average_treatment_effect = np.mean(target_group['Outcomes'].to_numpy()) - np.mean(source_group['Outcomes'].to_numpy())
        weight=est_lamda[i]/np.sum(est_lamda)
        est_theta[j] += average_treatment_effect* weight
    
print("thetaの平均値：",np.mean(est_theta))
print("thetaのMSE：",(np.mean(est_theta-true_values))**2)
print("thetaのbias：",np.mean(est_theta-true_values))
print("thetaのvaruance：",np.var(est_theta))
print("thetaのsd：",np.std(est_theta))
print("thetaの推定値：",est_theta)



