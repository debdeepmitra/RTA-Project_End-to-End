import pandas as pd
df=pd.read_csv(r'C:\Users\Debdeep\Desktop\TMLC\Project-1\RTA_Dataset.csv')

df.head()

df.shape

df.info()

len(df.iloc[0])

df.describe(include=['O']).T

df.isnull().sum()

# converting 'time' to datetime
df['Time'] = pd.to_datetime(df['Time'])

# date (day-month-year) time
df["Time"].dt.hour

# extracting hour and minute from timestamp
df['hour'] = df['Time'].dt.hour
df['minute'] = df['Time'].dt.minute
df.drop('Time', axis=1, inplace=True)

"""We are dropping columns that have a high percentage of NULL values (~50%). However, we will keep the column "Service_year_of_vehicle" as it may provide some insight for the cause of the accident!"""

df.drop(columns = ['Defect_of_vehicle', 'Vehicle_driver_relation', 'Work_of_casuality', 'Fitness_of_casuality'], inplace=True)

df.isnull().sum()

"""Next, we are filling the missing values in the columns using mode!"""

df['Educational_level'].fillna(df['Educational_level'].mode()[0], inplace=True)
df['Driving_experience'].fillna(df['Driving_experience'].mode()[0], inplace=True)
df['Type_of_vehicle'].fillna(df['Type_of_vehicle'].mode()[0], inplace=True)
df['Owner_of_vehicle'].fillna(df['Owner_of_vehicle'].mode()[0], inplace=True)
df['Service_year_of_vehicle'].fillna(df['Service_year_of_vehicle'].mode()[0], inplace=True)
df['Area_accident_occured'].fillna(df['Area_accident_occured'].mode()[0], inplace=True)
df['Lanes_or_Medians'].fillna(df['Lanes_or_Medians'].mode()[0], inplace=True)
df['Road_allignment'].fillna(df['Road_allignment'].mode()[0], inplace=True)
df['Types_of_Junction'].fillna(df['Types_of_Junction'].mode()[0], inplace=True)
df['Road_surface_type'].fillna(df['Road_surface_type'].mode()[0], inplace=True)
df['Type_of_collision'].fillna(df['Type_of_collision'].mode()[0], inplace=True)
df['Vehicle_movement'].fillna(df['Vehicle_movement'].mode()[0], inplace=True)

df.isnull().sum()


col_map={
    'Day_of_week': 'day_of_week',
    'Age_band_of_driver': 'driver_age',
    'Sex_of_driver': 'driver_sex',
    'Educational_level': 'educational_level',
    'Driving_experience': 'driving_experience',
    'Type_of_vehicle': 'vehicle_type',
    'Owner_of_vehicle': 'vehicle_owner',
    'Service_year_of_vehicle': 'service_year',
    'Area_accident_occured': 'accident_area',
    'Lanes_or_Medians': 'lanes',
    'Road_allignment': 'road_allignment',
    'Types_of_Junction': 'junction_type',
    'Road_surface_type': 'surface_type',
    'Road_surface_conditions': 'road_surface_conditions',
    'Light_conditions': 'light_condition',
    'Weather_conditions': 'weather_condition',
    'Type_of_collision': 'collision_type',
    'Number_of_vehicles_involved': 'vehicles_involved',
    'Number_of_casualties': 'casualties',
    'Vehicle_movement': 'vehicle_movement',
    'Casualty_class': 'casualty_class',
    'Sex_of_casualty': 'casualty_sex' ,
    'Age_band_of_casualty': 'casualty_age',
    'Casualty_severity': 'casualty_severity',
    'Pedestrian_movement': 'pedestrian_movement',
    'Cause_of_accident': 'accident_cause',
    'Accident_severity': 'accident_severity'
}
df.rename(columns=col_map, inplace=True)

for i in df.columns:
    print(f"Unique value in {i}:")
    print(df[i].unique(),'\n')

import numpy as np

def ordinal_encoder(df, feats):
    for feat in feats:
        feat_val = list(np.arange(df[feat].nunique()))
        feat_key = list(df[feat].sort_values().unique())
        feat_dict = dict(zip(feat_key, feat_val))
        df[feat] = df[feat].map(feat_dict)
    return df

df = ordinal_encoder(df, df.drop(['accident_severity'], axis=1).columns)
df.shape

df.head()

df['accident_severity'].value_counts()

"""We observer there is class imbalance in the dataset. We will use SMOTE to do the upsampling."""

from sklearn.model_selection import train_test_split, GridSearchCV

X = df.drop(['accident_severity', 'casualty_severity'], axis=1)
y = df['accident_severity']

#X = df[['hour', 'day_of_week', 'casualties', 'accident_cause', 'vehicles_involved', 'vehicle_type', 'driver_age', 'accident_area', 'driving_experience', 'lanes']]
#y = df['accident_severity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from imblearn.over_sampling import SMOTE

oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

y_train.value_counts()

y_test = ordinal_encoder(pd.DataFrame(y_test, columns = ['accident_severity']), pd.DataFrame(y_test, columns = ['accident_severity']).columns)['accident_severity']
y_train = ordinal_encoder(pd.DataFrame(y_train, columns = ['accident_severity']), pd.DataFrame(y_train, columns = ['accident_severity']).columns)['accident_severity']

y_train.value_counts()



from sklearn.metrics import (accuracy_score,
                            classification_report,
                            recall_score, precision_score, f1_score,
                            confusion_matrix)

from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

print(X_train.shape[1])
print(type(X_train))

scores={}

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
scores['xgb']= [accuracy_score(y_test, y_pred)]

rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
scores['rf']= [accuracy_score(y_test, y_pred)]

extree = ExtraTreesClassifier()
extree.fit(X_train, y_train)
y_pred = extree.predict(X_test)
scores['extree'] = [accuracy_score(y_test, y_pred)]

extree_tuned = ExtraTreesClassifier(ccp_alpha = 0.0,
                                criterion = 'gini',
                                min_samples_split = 3,
                                class_weight = 'balanced',
                                max_depth = 11,
                                n_estimators = 300)

extree_tuned.fit(X_train, y_train)
y_pred = extree.predict(X_test)
scores['extree_tuned'] = [accuracy_score(y_test, y_pred)]


print('XGB : ',scores['xgb'])
print('Random Forest : ',scores['rf'])
print('Extree: ',scores['extree'])
print('Extree_tuned: ',scores['extree_tuned'])


"""**Saving the model**"""

import joblib

joblib.dump(xgb, 'xgb.joblib')
joblib.dump(rf, 'rf.joblib')
joblib.dump(extree, 'extree.joblib')
joblib.dump(extree_tuned, 'extree_tuned.joblib')

'''
model_loaded = joblib.load('xgb.joblib')

answers = model_loaded.predict(X_test)

def mapping(answers):
  ans = []
  for val in answers:
    if val == 2:
      ans.append('Slight Injury')
    elif val == 1:
      ans.append('Serious Injury')
    elif val == 0:
      ans.append('Fatal Injury')

  return ans

result = mapping(answers)
print(result)
'''