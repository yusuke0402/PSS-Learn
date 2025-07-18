import numpy as np
import yaml
import random
from data import DataSets
from propensityscore import propensityscore
from trim import target_trim,source_trim

#0.初期設定
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
split=config["hyperparam"]["n_split"]
est_theta=np.zeros(config["hyperparam"]["n_trial"])
true_values=np.ones_like(est_theta)
for j in range(0,config["hyperparam"]["n_trial"]):
    #1.データ作成
    random.seed(j)
    np.random.seed(j)

    data=DataSets()   

    #2.傾向スコアの推定
    ppscore_target,ppscore_source = propensityscore(target_x=data.training_current_x[:,1:],target_y=data.training_current_y,source_x=data.training_historical_x[:,1:],source_y=data.training_historical_y)

    #3.傾向スコアに基づいてデータを層別化・トリミング
    target_data,split_values=target_trim(data.training_current_x,data.training_current_y.reshape(-1),ppscore_target)
    source_data=source_trim(data.training_historical_x,data.training_historical_y.reshape(-1),ppscore_source,split_values)

    #4.層ごとに平均治療効果を推定
    for i in range(0,split):
        target_condition = target_data['Group'] == i+1
        source_condition = source_data['Group'] == i+1  
        target_group=target_data[target_condition]
        source_group=source_data[source_condition]
        average_treatment_effect = np.mean(target_group['Outcomes'].to_numpy()) - np.mean(source_group['Outcomes'].to_numpy())
        weight=(len(source_group['Outcomes'])+len(target_group['Outcomes']) )/ (len(source_data['Outcomes'])+len(target_data['Outcomes']))
        est_theta[j] += average_treatment_effect* weight
    
print("thetaの平均値：",np.mean(est_theta))
print("thetaのMSE：",(np.mean(est_theta-true_values))**2)
print("thetaのbias：",np.mean(est_theta-true_values))
print("thetaのvaruance：",np.var(est_theta))
print("thetaのsd：",np.std(est_theta))
print("thetaの推定値：",est_theta)



