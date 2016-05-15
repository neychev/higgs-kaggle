import numpy as np
import pandas as pd
import  xgboost as xgb
import matplotlib as mplt
from __future__ import division

def read_data(train_path = 'data/training.csv', test_path = 'data/test.csv'):
    data_train = pd.read_csv(train_path)
    data_test = pd.read_csv(test_path)
    data_train['Label'] = data_train['Label'].map(lambda x: 1 if x == 's' else (0 if x == 'b' else -1))
    return data_train, data_test

data_train, data_test = read_data()


for x in list(data_train.columns):
    tmp = data_train[x].map(lambda  y: None if y == -999.0 else y)
    nan_num = tmp.isnull().sum()
    if (nan_num != 0):
        print(x)


weights = data_train.Weight.as_matrix()
indexes_train = data_train.EventId.as_matrix()
indexes_test = data_test.EventId.as_matrix()
y_train = data_train.Label.as_matrix()
X_train = data_train.ix[:, 1:-1-1].as_matrix()
X_test =  data_test.ix[:, 1:].as_matrix()

#compute 'scale_posweigth' for xgboost model:
twr = weights[y_train == 1].sum()
fwr = weights[y_train == 0].sum()

# tmp = weights[y_train == 1].sum() + weights[y_train == 0].sum()
# weights.sum() - tmp

xgmat = xgb.DMatrix(X_train, label=y_train, missing = -999.0, weight=weights)

#Imported from tqchen git
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = fwr/twr
param['eta'] = 0.1
param['max_depth'] = 6
param['eval_metric'] = 'ams@0.15'
param['silent'] = 1
plst = list(param.items())

num_rounds = 250
bst = xgb.train(plst, xgmat, num_rounds)
print('Training finished')


xgmat_pred = xgb.DMatrix(X_test, missing=-999.0)
y_pred_proba = bst.predict(xgmat_pred)
print('Prediction computed')


threshold_ratio = 0.15
outfile = "results(xgb).csv"

res = [(int(indexes_test[i]), y_pred_proba[i]) for i in range(len(y_pred_proba))]


rorder = {}
rorder_len = 0
for k, v in sorted(res, key = lambda x: x[1], reverse=True):
    rorder[k] = rorder_len + 1
    rorder_len += 1


ntop = int(threshold_ratio * rorder_len)
output = open(outfile, 'w')
nhit = 0
ntot = 0
output.write('EventId,RankOrder,Class\n')
for k, tmp in res:
    if rorder[k] <= ntop:
        lb = 's'
        nhit += 1
    else:
        lb = 'b'

    output.write('%s,%d,%s\n' % (k,  len(rorder) + 1 - rorder[k], lb))
    ntot += 1
output.close()



feature_names = list(data_test[1:].columns)
for feature in feature_names:
    tmp = data_train.DER_mass_MMC[data_train.DER_mass_MMC != -999.0]
    tmp.hist(bins=100)
    mplt.xlabel(feature)
    mplt.show()