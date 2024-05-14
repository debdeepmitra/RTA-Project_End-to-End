import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

def ordinal_encoder(input_val, feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value

def get_prediction(data, model):
    val = model.predict(data)
    if val == 2:
      return 'Slight Injury'
    elif val == 1:
      return 'Serious Injury'
    elif val == 0:
     return 'Fatal Injury'