import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from xgboost import XGBRFClassifier
from lightgbm.sklearn import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class conf:
    index = 'id'
    target = 'stroke'
    random = 2023

    folds = 12

np.random.seed(conf.random)

# 1. READ DATA
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", None)

train_path = "data/train.csv"
train_dataset = pd.read_csv(train_path, index_col=conf.index)

test_path = "data/test.csv"
test_dataset = pd.read_csv(test_path, index_col=conf.index)

# 2. DATA ANALYSIS

# general information
print(f'\nThe training set contains {train_dataset.shape[0]} rows and {train_dataset.shape[1]} columns.\n')

print("First 5 rows: ", end="\n\n")
print(train_dataset.head(), end="\n\n")

print("Last 5 rows:", end="\n\n")
print(train_dataset.tail(), end="\n\n")

print("Info about train_dataset:", end="\n\n")
print(train_dataset.info(), end='\n\n')  # data profiling

print("General statistic information about attributes:", end='\n\n')
print(train_dataset.describe(), end='\n\n')  # feature statistic

print("Number of missing values for each column:", end='\n\n')
print(train_dataset.isna().sum(), end='\n\n')

# attributes

# gender
print("Gender types:")
print(train_dataset['gender'].unique(), end='\n\n')

# gender count
print("Gender counts:")
print(pd.crosstab(
    index=train_dataset['gender'],
    columns='counts'
).T, end='\n\n')

# gender count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="gender", width=0.5, hue='gender', legend=True, palette='rocket')
ax.set_title('gender count', fontsize=16, fontweight="bold")
plt.savefig('gender_count.png', dpi=250)
plt.close()

# How gender effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='gender', hue='stroke', legend=True, palette='rocket')
ax.set_title('gender vs stroke', fontsize=16, fontweight="bold")
plt.savefig('gender_vs_stroke.png', dpi=250)
plt.close()

# gender vs stroke
print("gender vs stroke:")
print(pd.crosstab(
    index=train_dataset['gender'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# age
print("Age information:")
print(train_dataset['age'].describe(), end='\n\n')

# Age distribution visualization
plt.figure(figsize=(8, 6))
ax = sns.distplot(train_dataset['age'])
ax.set_title('age distribution', fontsize=16, fontweight="bold")
plt.savefig('age_distribution.png', dpi=250)
plt.close()

# How age feature effects the target variable
plt.figure(figsize=(10, 6))
ax = sns.boxenplot(data=train_dataset, x='stroke', y='age', hue='stroke', palette='rocket')
ax.set_title('age vs stroke', fontsize=16, fontweight="bold")
plt.savefig('age_vs_stroke.png', dpi=250)
plt.close()

# hypertension
print("hypertension types:")
print(train_dataset['hypertension'].unique(), end='\n\n')

# hypertension count
print("Hypertension counts:")
print(pd.crosstab(
    index=train_dataset['hypertension'],
    columns='counts'
).T, end='\n\n')

# hypertension count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="hypertension", width=0.5, hue='hypertension', legend=True, palette='rocket')
ax.set_title('hypertension count', fontsize=16, fontweight="bold")
plt.savefig('hypertension_count.png', dpi=250)
plt.close()

# How hypertension feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='hypertension', hue='stroke', legend=True, palette='rocket')
ax.set_title('hypertension vs stroke', fontsize=16, fontweight="bold")
plt.savefig('hypertension_vs_stroke.png', dpi=250)
plt.close()

# Hypertension vs stroke
print("hypertension vs stroke:")
print(pd.crosstab(
    index=train_dataset['hypertension'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# heart_disease
print("Heart disease types:")
print(train_dataset['heart_disease'].unique(), end='\n\n')

# heart_disease count
print("Heart disease counts:")
print(pd.crosstab(
    index=train_dataset['heart_disease'],
    columns='counts'
).T, end='\n\n')

# heart_disease count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="heart_disease", width=0.5, hue='heart_disease', legend=True, palette='rocket')
ax.set_title('heart_disease count', fontsize=16, fontweight="bold")
plt.savefig('heart_disease_count.png', dpi=250)
plt.close()

# How heart_disease feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='heart_disease', hue='stroke', legend=True, palette='rocket')
ax.set_title('heart_disease vs stroke', fontsize=16, fontweight="bold")
plt.savefig('heart_disease_vs_stroke.png', dpi=250)
plt.close()

# heart_disease vs stroke
print("heart_disease vs stroke:")
print(pd.crosstab(
    index=train_dataset['heart_disease'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# ever_married
print("Ever married types:")
print(train_dataset['ever_married'].unique(), end='\n\n')

# ever_married count
print("Ever married counts:")
print(pd.crosstab(
    index=train_dataset['ever_married'],
    columns='counts'
).T, end='\n\n')

# ever_married count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="ever_married", width=0.5, hue='ever_married', legend=True, palette='rocket')
ax.set_title('ever_married count', fontsize=16, fontweight="bold")
plt.savefig('ever_married_count.png', dpi=250)
plt.close()

# How ever_married feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='ever_married', hue='stroke', legend=True, palette='rocket')
ax.set_title('ever_married vs stroke', fontsize=16, fontweight="bold")
plt.savefig('ever_married_vs_stroke.png', dpi=250)
plt.close()

# Ever Married vs stroke
print("ever_married vs stroke:")
print(pd.crosstab(
    index=train_dataset['ever_married'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# work_type
print("Work types:")
print(train_dataset['work_type'].unique(), end='\n\n')

# work_type count
print("Work type counts:")
print(pd.crosstab(
    index=train_dataset['work_type'],
    columns='counts'
).T, end='\n\n')

# work_type count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="work_type", width=0.5, hue='work_type', legend=True, palette='rocket')
ax.set_title('work_type count', fontsize=16, fontweight="bold")
plt.savefig('work_type_count.png', dpi=250)
plt.close()

# How work_type feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='work_type', hue='stroke', legend=True, palette='rocket')
ax.set_title('work_type vs stroke', fontsize=16, fontweight="bold")
plt.savefig('work_type_vs_stroke.png', dpi=250)
plt.close()

# Work Type vs stroke
print("work_type vs stroke:")
print(pd.crosstab(
    index=train_dataset['work_type'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# Residence_type
print("Residence types:")
print(train_dataset['Residence_type'].unique(), end='\n\n')

# work_type count
print("Residence type counts:")
print(pd.crosstab(
    index=train_dataset['Residence_type'],
    columns='counts'
).T, end='\n\n')

# Residence_type count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="Residence_type", width=0.5, hue='Residence_type', legend=True,
                   palette='rocket')
ax.figure.get_axes()[0].legend(title='loss', loc='upper right')
ax.set_title('Residence_type count', fontsize=16, fontweight="bold")
plt.savefig('Residence_type_count.png', dpi=250)
plt.close()

# How Residence_type feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='Residence_type', hue='stroke', legend=True, palette='rocket')
ax.set_title('Residence_type vs stroke', fontsize=16, fontweight="bold")
plt.savefig('residence_type_vs_stroke.png', dpi=250)
plt.close()

# Residence_type vs stroke
print("Residence_type vs stroke:")
print(pd.crosstab(
    index=train_dataset['Residence_type'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# avg_glucose_level
print("Average glucose level information:")
print(train_dataset['avg_glucose_level'].describe(), end='\n\n')

# avg_glucose_level distribution visualization
plt.figure(figsize=(8, 6))
ax = sns.distplot(train_dataset['avg_glucose_level'])
ax.set_title('avg_glucose_level distribution', fontsize=16, fontweight="bold")
plt.savefig('avg_glucose_level_distribution.png', dpi=250)
plt.close()

# How avg_glucose_level feature effects the target variable
plt.figure(figsize=(10, 6))
ax = sns.boxenplot(data=train_dataset, x='stroke', y='avg_glucose_level', hue='stroke', palette='rocket')
ax.set_title('avg_glucose_level vs stroke', fontsize=16, fontweight="bold")
plt.savefig('avg_glucose_level_vs_stroke.png', dpi=250)
plt.close()

# bmi
print("Bmi information:")
print(train_dataset['bmi'].describe(), end='\n\n')

# bmi distribution visualization
plt.figure(figsize=(8, 6))
ax = sns.distplot(train_dataset['bmi'])
ax.set_title('bmi distribution', fontsize=16, fontweight="bold")
plt.savefig('bmi_distribution.png', dpi=250)
plt.close()

# How bmi feature effects the target variable
plt.figure(figsize=(10, 6))
ax = sns.boxenplot(data=train_dataset, x='stroke', y='bmi', hue='stroke', palette='rocket')
ax.set_title('bmi vs stroke', fontsize=16, fontweight="bold")
plt.savefig('bmi_vs_stroke.png', dpi=250)
plt.close()

# smoking_status
print("Smoking status types:")
print(train_dataset['smoking_status'].unique(), end='\n\n')

# smoking_status count
print("Smoking status counts:")
print(pd.crosstab(
    index=train_dataset['smoking_status'],
    columns='counts'
).T, end='\n\n')

# smoking_status count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="smoking_status", width=0.5, hue='smoking_status', legend=True,
                   palette='rocket')
ax.set_title('smoking status count', fontsize=16, fontweight="bold")
plt.savefig('smoking_status_count.png', dpi=250)
plt.close()

# How smoking_status feature effects the target variable
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x='smoking_status', hue='stroke', legend=True, palette='rocket')
ax.set_title('smoking_status vs stroke', fontsize=16, fontweight="bold")
plt.savefig('smoking_status_vs_stroke.png', dpi=250)
plt.close()

# smoking_status vs stroke
print("smoking_status vs stroke:")
print(pd.crosstab(
    index=train_dataset['smoking_status'],
    columns=train_dataset['stroke'],
    margins=True,
    normalize='index'
), end='\n\n')

# stroke - target
print("stroke types:")
print(train_dataset['stroke'].unique(), end='\n\n')

# stroke count
print("stroke counts:")
print(pd.crosstab(
    index=train_dataset['stroke'],
    columns='counts'
).T, end='\n\n')

# stroke count visualization
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=train_dataset, x="stroke", width=0.5, hue='stroke', legend=True, palette='rocket')
ax.set_title('The target variable count', fontsize=16, fontweight="bold")
plt.savefig('stroke_count.png', dpi=250)
plt.close()


# 3. FEATURE ENGINEERING

# load the original dataset
df_orig = pd.read_csv('data/healthcare-dataset-stroke-data.csv', index_col=0)

# combine it with the training set
df_orig_stroke_1 = df_orig[df_orig[['stroke']].all(1)]
train_dataset = pd.concat([train_dataset, df_orig_stroke_1])

# combine the training and the test sets for consistency purposes
df = pd.concat([train_dataset, test_dataset])

# reset id since the original dataset also begins with 0
df.index = df.reset_index().index

# remove 'Residence_type' and 'bmi' features
df.drop(['Residence_type', 'bmi'], axis=1, inplace=True)

# separate the training set and the test set from each other.
df_train = df.loc[:train_dataset.shape[0] - 1]
df_test = df.loc[df_train.shape[0]:].iloc[:, :-1]

# separate the features from the target variable
X = df_train.iloc[:, :-1]
y = df_train.pop(conf.target).to_numpy()

num_cols = ["age", "avg_glucose_level"]
cat_cols = X.columns.difference(num_cols)

num_pipe = Pipeline([
    ('scaler', MinMaxScaler())
])

tr = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", OrdinalEncoder(), cat_cols)
])

X = tr.fit_transform(X)
df_test = tr.transform(df_test)
print("train shape = ", X.shape)
print("test shape = ", df_test.shape)

models = []
skf = RepeatedKFold(n_splits=conf.folds, random_state=conf.random)

models = {
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=125),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=50, min_samples_leaf=50, max_depth=10,
                                                     max_samples=None, class_weight='balanced', random_state=2023),
    'XGBRFClassifier': XGBRFClassifier(n_estimators=1000),
    'LGBMClassifier': LGBMClassifier(n_estimators=24, random_state=2023),
    'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=10000)
}

print("\n------------------------------ TRAIN RESULTS ------------------------------")
pd.set_option('display.max_colwidth', None)
pd.set_option('display.colheader_justify', 'center')

results = pd.DataFrame(columns=['roc_auc_score'])
for model_name, model in models.items():
    scores = []
    for train_index, val_index in skf.split(X, y):
        x_train, x_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(x_train, y_train)
        y_proba = model.predict_proba(x_val)
        scores.append(roc_auc_score(y_val, y_proba[:, 1]))

    results.loc[model_name] = np.mean(scores)

print("\n", results)
