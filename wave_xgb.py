
import xgboost
import shap
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import math
import time
import seaborn as sns
import scipy
import os
import imgkit
import pickle

## =========================== Part 1: Loading Data ===========================

start= time.time()
input1 = pd.read_csv("inputdata.csv", names = [r'$KC$',r'$Red$',r'$Shields$', r'$Rep$'])
output1 = pd.read_csv("outputdata.csv", names = [r'$S/D$'])


# dataset loading
value_input = input1.values
value_output = output1.values
input_length = len(value_input)

shaped_input = value_input.reshape(input_length,4)
shaped_output = value_output.reshape(input_length, 1)

#初始化结果图的文件夹
folder = "visualization"
if not os.path.exists(folder):
    os.makedirs(folder)

if not os.path.exists("pkl"):
    os.makedirs("pkl")

## ====================== Part 2: Creating Train and Test Sets ======================

X_train, X_test, y_train, y_test = train_test_split(shaped_input, shaped_output ,test_size=0.2, random_state=42)

# 将训练集和测试集数据保存到 CSV 文件中
train_data = pd.DataFrame(X_train, columns=input1.columns)
train_data['Target'] = y_train
test_data = pd.DataFrame(X_test, columns=input1.columns)
test_data['Target'] = y_test

# 保存文件
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)


## ====================== Part 3: Defining XGBoost Model Parameters and Training XGBoost Model ======================
params = {'objective': 'reg:squarederror',
          'base_score': 1,
          'booster': 'gbtree',
          'colsample_bylevel': 1,
          'colsample_bytree': 1,
          'n_estimators': 800,
          'learning_rate': 0.03,
          'gamma':0,
          'subsample':0.7,
          'max_depth':6,
          'min_child_weight':8,
          'scale_pos_weight':1,
          'reg_alpha': 0.1,
          }

xgb_model = xgboost.XGBRegressor(**params)
xgb_model.fit(X_train, y_train, early_stopping_rounds=50, eval_metric='rmse',  eval_set=[(X_test, y_test)], verbose=False)
mse = mean_squared_error(y_test, xgb_model.predict(X_test))


## ====================== Part 4: Metrics and Analysis ======================
Y_prediction_test= xgb_model.predict(X_test) ## Prediction of Test set
Y_prediction_train = xgb_model.predict(X_train) ## Prediction of Train set
Y_total = xgb_model.predict(input1) ## Prediction of Totall

print("time :", time.time() - start)

test = y_test ## Measured value(Test set)
prediction_test = Y_prediction_test ## Predicted value (Test set)
train = y_train ## Measured value (Training set)
prediction_train = Y_prediction_train ## Predicted value (Training set)


test = test.flatten()
prediction_test = prediction_test.flatten()
train = train.flatten()
prediction_train = prediction_train.flatten()

Mean_test = np.mean(test)
Mean_prediction_test = np.mean(prediction_test)
Mean_train = np.mean(prediction_train)
Mean_prediction_train = np.mean(train)


'''
## RMSE
RMSE_test = math.sqrt(sum([(prediction_test[i]-test[i])**2/len(test) for i in range(len(test))]))
print("RMSE_test: %f" % RMSE_test)
RMSE_train = math.sqrt(sum([(train[i]-prediction_train[i])**2/len(train) for i in range(len(train))]))
print("RMSE_train: %f" % RMSE_train)

## NMSE
NMSE_test = sum([(prediction_test[i]-test[i])**2/(Mean_test*Mean_prediction_test*len(test)) for i in range(len(test))])
print("NMSE_test: %f" % NMSE_test)
NMSE_train = math.sqrt(sum([(train[i]-prediction_train[i])**2/(Mean_train*Mean_prediction_train*len(train)) for i in range(len(train))]))
print("NMSE_train: %f" % NMSE_train)

## I
divisor = sum([(prediction_test[i]-test[i])**2 for i in range(len(test))])
dividend = sum([(abs(test[i]-Mean_test)+abs(prediction_test[i]-Mean_test))**2 for i in range(len(test))])
Ia_test= 1-divisor/dividend
print("Itest: %f" % Ia_test)

divisor = sum([(prediction_train[i]-train[i]) **2 for i in range(len(train))])
dividend = sum([(abs(train[i]-Mean_train)+abs(prediction_train[i]-Mean_train))**2 for i in range(len(train))])
Ia_train= 1-divisor/dividend
print("Itrain: %f" % Ia_train)

## SI
divisor = math.sqrt(sum([(prediction_test[i]-test[i])**2/len(test) for i in range(len(test))]))
SI_test = divisor/Mean_test
print("SI_test: %f" % SI_test)
divisor = math.sqrt(sum([(train[i]-prediction_train[i])**2/len(train) for i in range(len(train))]))
SI_train = divisor/Mean_prediction_train
print("SI_train: %f" % SI_train)

## NSE
divisor = sum([(test[i]-prediction_test[i]) **2 for i in range(len(test))])
dividend = sum([(test[i]-Mean_test) **2 for i in range(len(test))])
nse_test= 1-divisor/dividend
print("nse_test: %f" % nse_test)

divisor = sum([(prediction_train[i]-train[i]) **2 for i in range(len(train))])
dividend = sum([(train[i]-Mean_prediction_train) **2 for i in range(len(train))])
nse_train= 1-divisor/dividend
print("nse_train: %f" % nse_train)

## R2
a1 = sum([(test[i]-Mean_test)*(prediction_test[i]-Mean_prediction_test) for i in range(len(test))])
a2 = math.sqrt(sum([(test[i]-Mean_test)**2 for i in range(len(test))]))
a3 = math.sqrt(sum([(prediction_test[i]-Mean_prediction_test)**2 for i in range(len(test))]))
r2_test= (a1/(a2*a3))**2
print("r_test: %f" % r2_test)

a1 = sum([(train[i]-Mean_train)*(prediction_train[i]-Mean_prediction_train) for i in range(len(train))])
a2 = math.sqrt(sum([(train[i]-Mean_train)**2 for i in range(len(train))]))
a3 = math.sqrt(sum([(prediction_train[i]-Mean_prediction_train)**2 for i in range(len(train))]))
r2_train= (a1/(a2*a3))**2
print("r_train: %f" % r2_train)


## B
B_test = sum([(prediction_test[i]-test[i])/len(test) for i in range(len(test))])
print("B_test: %f" % B_test)

B_train = sum([(prediction_train[i]-train[i])/len(prediction_train) for i in range(len(prediction_train))])
print("B_train: %f" % B_train)

## Se
Se_test = math.sqrt(sum([((prediction_test[i]-test[i])-B_test)**2/(len(test)-2) for i in range(len(test))]))
print("SE_test: %f" % Se_test)
Se = math.sqrt(sum([((prediction_train[i]-train[i])-B_train)**2/(len(prediction_train)-2) for i in range(len(prediction_train))]))
print("SE_train: %f" % Se)


# 将错误和性能指标存储在字典中
error_metrics = {
    "Metric": [
        "RMSE_test", "RMSE_train",
        "NMSE_test", "NMSE_train",
        "I_test", "I_train",
        "SI_test", "SI_train",
        "NSE_test", "NSE_train",
        "R2_test", "R2_train",
        "B_test", "B_train",
        "SE_test", "SE_train"
    ],
    "Value": [
        RMSE_test, RMSE_train,
        NMSE_test, NMSE_train,
        Ia_test, Ia_train,
        SI_test, SI_train,
        nse_test, nse_train,
        r2_test, r2_train,
        B_test, B_train,
        Se_test, Se
    ]
}

# 创建 DataFrame
error_df = pd.DataFrame(error_metrics)

# 保存 DataFrame 到 CSV 文件
error_df.to_csv("error.csv", index=False)
'''
# 以下函数计算各种性能指标
def calculate_rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def calculate_nmse(predictions, targets):
    return ((predictions - targets) ** 2).sum() / ((targets - np.mean(targets)) ** 2).sum()

def calculate_index_of_agreement(predictions, targets):
    denominator = ((abs(targets - np.mean(targets)) + abs(predictions - np.mean(targets))) ** 2).sum()
    numerator = ((predictions - targets) ** 2).sum()
    return 1 - numerator / denominator

def calculate_si(predictions, targets):
    rmse = calculate_rmse(predictions, targets)
    return rmse / np.mean(targets)

def calculate_nse(predictions, targets):
    return 1 - ((predictions - targets) ** 2).sum() / ((targets - np.mean(targets)) ** 2).sum()

def calculate_r2(predictions, targets):
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - np.mean(targets)) ** 2).sum()
    return 1 - ss_res / ss_tot

def calculate_bias(predictions, targets):
    return (predictions - targets).mean()

def calculate_se(predictions, targets, bias):
    n = len(targets)
    return np.sqrt((((predictions - targets) - bias) ** 2).sum() / (n - 2))

def calculate_metrics(predictions, targets):
    rmse = calculate_rmse(predictions, targets)
    nmse = calculate_nmse(predictions, targets)
    i_agree = calculate_index_of_agreement(predictions, targets)
    si = calculate_si(predictions, targets)
    nse = calculate_nse(predictions, targets)
    r2 = calculate_r2(predictions, targets)
    bias = calculate_bias(predictions, targets)
    se = calculate_se(predictions, targets, bias)
    mean_actual = np.mean(targets)
    mean_predicted = np.mean(predictions)
    return {
        "RMSE": rmse, 
        "NMSE": nmse, 
        "I": i_agree, 
        "SI": si, 
        "NSE": nse, 
        "R2": r2, 
        "Bias": bias, 
        "SE": se,
        "Mean_Actual": mean_actual, 
        "Mean_Predicted": mean_predicted
    }


# 计算训练集和测试集的指标
metrics_train = calculate_metrics(prediction_train, y_train)
metrics_test = calculate_metrics(prediction_test, y_test)

# 创建DataFrame并保存为CSV
metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "NMSE", "I", "SI", "NSE", "R2", "Bias", "SE", "Mean_Actual", "Mean_Predicted"],
    "Train": [metrics_train[key] for key in ["RMSE", "NMSE", "I", "SI", "NSE", "R2", "Bias", "SE", "Mean_Actual", "Mean_Predicted"]],
    "Test": [metrics_test[key] for key in ["RMSE", "NMSE", "I", "SI", "NSE", "R2", "Bias", "SE", "Mean_Actual", "Mean_Predicted"]]
})
metrics_df.to_csv("model_metrics.csv", index=False)

# Measured versus Predicted
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 计算置信区间
std_error = np.std(prediction_test - test) / np.sqrt(len(test))
lower_bound = (test - 1.645 * std_error)
upper_bound = (test + 1.645 * std_error)

# Test
axs[0].plot(test, prediction_test, 'o', label='Predicted Data', markersize=8, alpha=0.7)
axs[0].plot(test, test, color='red', linestyle='dashed', label='1:1 Line', linewidth=2)
axs[0].plot(test, lower_bound, 'r-', label='Lower 90% CI', linewidth=1, alpha=0.8)
axs[0].plot(test, upper_bound, 'r-', label='Upper 90% CI', linewidth=1, alpha=0.8)
axs[0].set_xlabel('Oveserved Values', fontsize=12)
axs[0].set_ylabel('Predicted Values', fontsize=12)
axs[0].set_title('Test Data Prediction', fontsize=14)
axs[0].legend()
axs[0].grid(True, linestyle='--', alpha=0.5)

# Training
axs[1].plot(train, prediction_train, 'o', label='Predicted Data', markersize=8, alpha=0.7)
axs[1].plot(train, train, color='red', linestyle='dashed', label='1:1 Line', linewidth=2)
axs[1].set_xlabel('Oveserved Values', fontsize=12)
axs[1].set_ylabel('Predicted Values', fontsize=12)
axs[1].set_title('Train Data Prediction', fontsize=14)
axs[1].legend()
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()  # Adjust layout for better spacing

# Save the subplots
# plt.show()
plt.savefig(f"{folder}/prediction.png")
plt.clf()

# Error distribution
Error_test = prediction_test- test
Error_train = prediction_train- train


## Error distribution

# Error distribution for Test set
fig = plt.figure(figsize=(20, 10))
plt.subplot(121)
sns.distplot(
    Error_test,
    kde=False,
    fit=scipy.stats.norm,
    bins=15,
    hist_kws={
        "rwidth": 0.8,
        "color": "blue",
        "edgecolor": "black",
        "alpha": 0.7
    },
    fit_kws={
        "color": "red",
        "linestyle": "dashed"
    }
)
plt.xlim(-1.5, 1.5)
plt.xlabel('$y_{measured} - y_{predicted}$')
plt.ylabel("Normalized PDF")
plt.title("Error Distribution (Test Set)")
plt.grid(True)

# Error distribution for Training set
plt.subplot(122)  # Change the subplot number to 122
sns.distplot(
    Error_train,
    kde=False,
    fit=scipy.stats.norm,
    bins=15,
    hist_kws={
        "rwidth": 0.8,
        "color": "blue",
        "edgecolor": "black",
        "alpha": 0.7
    },
    fit_kws={
        "color": "red",
        "linestyle": "dashed"
    }
)
plt.xlim(-1.5, 1.5)
plt.xlabel('$y_{measured} - y_{predicted}$')
plt.ylabel("Normalized PDF")
plt.title("Error Distribution (Training Set)")
plt.grid(True)

plt.tight_layout()
# plt.show()
plt.savefig(f"{folder}/error_distribution.png")
plt.clf()

'''
## ====================== Part 5: Model Generalization ======================

#Johnes+sheppard
regulation= pd.read_csv("regulation_renew.csv", names = [r'$b/d_50$',r'$y/b$',r'$Fr$', r'$V/V_c$', r' $y_s/b$'])
X_regulation= regulation.loc [:, r'$b/d_50$': r'$V/V_c$']
Y_regulation_observed = regulation.loc [:,r' $y_s/b$']
Y_regulation_predicted = xgb_model.predict(X_regulation)


# Generalization Figure
X=np.linspace(0,3.5,100)
y1=X+ B_test
y2=X+(B_test+1.96*Se_test)
y3=X-(B_test+1.96*Se_test)

plt.figure(figsize=(10, 6))
plt.scatter(Y_regulation_observed, Y_regulation_predicted, color='blue', alpha=0.7)
plt.plot(X, y1, color='red', linestyle='solid')
plt.plot(X, y2, color='red', linestyle='dashed')
plt.plot(X, y3, color='red', linestyle='dotted')
plt.xlim(0, 3.5)
plt.ylim(0, 3.5)
plt.xlabel('Observed value($y_{so/b}$)', fontsize=12)
plt.ylabel('Predicted value($y_{sp/b}$)', fontsize=12)
plt.title('Observed vs. Predicted Values ', fontsize=14)
plt.legend(['Genralization', 'Mean Line of Prediction Error','Upper Line of 95% C.I.', 'Lower Line of 95% C.I.'])
plt.grid(True)
plt.tight_layout()
plt.show()

'''


## ====================== Part 6: Model Interpretation and Visualization ======================

# Initialize JavaScript for SHAP plot
shap.initjs()
# Create a SHAP TreeExplainer
explainer = shap.TreeExplainer(xgb_model)
# Calculate SHAP values
shap_values = explainer.shap_values(input1)

# Force plot for a single prediction
shap_html = shap.force_plot(explainer.expected_value, shap_values, input1, show=False)
shap.save_html(f"{folder}/force_plot.html", shap_html)

# Summary plot for feature importance
shap.summary_plot(shap_values, input1, show=False)
plt.savefig(f"{folder}/summary_plot.png")
plt.clf()

shap.summary_plot(shap_values, input1, plot_type="bar", color='blue', show=False)
plt.savefig(f"{folder}/summary_bar_plot.png")
plt.clf()

# Individual feature dependence plots
if not os.path.exists(f"{folder}/dependence_plot_feature"):
    os.makedirs(f"{folder}/dependence_plot_feature")
for feature_idx in range(input1.shape[1]):
    shap.dependence_plot(feature_idx, shap_values, input1, dot_size=45, show=False)
    plt.savefig(f"{folder}/dependence_plot_feature/dependence_plot_feature_{feature_idx}.png")
    plt.clf()

# Individual feature dependence plots without interaction_index
if not os.path.exists(f"{folder}/dependence_plot_feature_without_interaction_index"):
    os.makedirs(f"{folder}/dependence_plot_feature_without_interaction_index")
for feature_idx in range(input1.shape[1]):
    shap.dependence_plot(feature_idx, shap_values, input1, interaction_index=None, dot_size=45, show=False)
    plt.savefig(f"{folder}/dependence_plot_feature_without_interaction_index/dependence_plot_feature_without_interaction_index_{feature_idx}.png")
    plt.clf()


# 创建一个 SHAP 解释器
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(input1)

# heat map
shap.plots.heatmap(shap_values, show=False)
plt.savefig(f"{folder}/heatmap.png")
plt.clf()

# 计算离群点
residuals_test = prediction_test - test
std_dev = np.std(residuals_test)
outliers = np.where((residuals_test > 2 * std_dev) | (residuals_test < -2 * std_dev))[0]

def generate_waterfall_plots(explainer, shap_values, outliers, folder):
    for idx in outliers:
        # 生成瀑布图
        plt.figure()
        shap.plots.waterfall(shap_values[idx], show=False)  # max_display 控制显示最重要的特征数量
        
        # 保存图像
        plt.savefig(f"{folder}/waterfall_plot_index_{idx}.png")
        print(f"save{idx}")
        plt.clf()
        print(f"close{idx}")

print("waterfall")

# 调用函数生成并保存离群点的瀑布图
generate_waterfall_plots(explainer, shap_values, outliers, folder)


'''
# Example waterfall plots for specific cases 点出特殊值
explainer = shap.Explainer(xgb_model, input1)
shap_values = explainer(input1)
shap.plots.heatmap(shap_values)
shap.plots.waterfall(shap_values[261]) ##Case 1  b/d_50=3.67, V/V_c=0.95, y/b =20.95, Fr =0.50
shap.plots.waterfall(shap_values[385]) ##Case 2  b/d50=203.6, V/V_c=3.91, y/b =2.0, Fr=1
shap.plots.waterfall(shap_values[359]) ##Case 3-1 b'=0.33, V=1.15, y=1.97, d50=0.55, sediment nonuniformity= 4.6, ysm/b=0.45
shap.plots.waterfall(shap_values[362]) ##Case 3-2 b'=0.33, V=1.15, y=1.97, d50=0.55, sediment nonuniformity= 1.6, ysm/b=2.20
shap.plots.waterfall(shap_values[363]) ##Case 4-1 b'=0.33, V=1.38, y=1.97, d50=0.85, sediment nonuniformity= 3.3, ysm/b=0.40
shap.plots.waterfall(shap_values[367]) ##Case 4-2 b'=0.33, V=1.15, y=1.97, d50=0.85, sediment nonuniformity= 1.3, ysm/b=2.10
shap.plots.waterfall(shap_values[368]) ##Case 5-1 b'=0.33, V=1.15, y=1.97, d50=0.85, sediment nonuniformity= 1.3, ysm/b=2.10
'''

'''
# Force plot for a single prediction
shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(input1)
shap.force_plot(explainer.expected_value, shap_values[261], input1.iloc[261, :], matplotlib=True)
plt.savefig(f"{folder}/single_prediction_force_plot.png")
plt.clf()
'''


'''
## ====================== Part 7: Using model ======================

bd50_values = []
y_over_b_values = []
Fr_values = []
V_over_Vc_values = []

# Get user input for each feature
num_inputs = int(input("Enter the number of predictions you want to make: "))
for _ in range(num_inputs):
    bd50_values.append(float(input("Enter $b/d_{50}$ value: ")))
    y_over_b_values.append(float(input("Enter $y/b$ value: ")))
    Fr_values.append(float(input("Enter $Fr$ value: ")))
    V_over_Vc_values.append(float(input("Enter $V/V_c$ value: ")))

# Create a dictionary with the collected input data
input_data = {
    '$b/d_{50}$': bd50_values,
    '$y/b$': y_over_b_values,
    '$Fr$': Fr_values,
    '$V/V_c$': V_over_Vc_values
}

# Create a DataFrame from the input data
input_df = pd.DataFrame.from_dict(input_data)

# Make predictions using the loaded model
predicted_values = xgb_model.predict(input_df.values)
Revised_predicted_value= predicted_values*1.3

# Display the predicted values
for i, predicted_value in enumerate(predicted_values):
    print(f"Predicted Value {i + 1}: {predicted_value}")
for i, predicted_value in enumerate(predicted_values):
    print(f"Revised Predicted Value {i + 1}: {Revised_predicted_value}")


explainer = shap.Explainer(xgb_model, input1)
shap_values = explainer(input_df.values)
shap.plots.waterfall(shap_values[0])
'''


## ====================== Part 8: Export pkl ======================

# 保存XGBoost模型
with open(f"pkl/xgb_model.pkl", 'wb') as file:
    pickle.dump(xgb_model, file)

# 保存训练集和测试集数据
with open(f"pkl/train_test_data.pkl", 'wb') as file:
    pickle.dump((X_train, X_test, y_train, y_test), file)

# 保存SHAP值（假设已经计算）
with open(f"pkl/shap_values.pkl", 'wb') as file:
    pickle.dump(shap_values, file)
