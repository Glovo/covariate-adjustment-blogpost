import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

def get_ate_cuped(df, prediction_column="Y_before"):
    theta = df.cov().loc[prediction_column, "Y"] / df.cov().loc[prediction_column, "Y"]
    
    r = df.copy()
    r['Y_cuped'] = df['Y'] - theta * (df[prediction_column] - df[prediction_column].mean())
    
    return r.loc[r['T']==1, 'Y_cuped'].mean() - r.loc[r['T']==0, 'Y_cuped'].mean()

def get_ate_ols(df, covariates=None):
    covariates = [] if not covariates else covariates
    T = 'T +' if covariates else 'T'
    result = smf.ols(f'Y ~ {T}' + '+'.join(covariates), data=df).fit()
    return result.params['T']

def add_cupac_predictions(pre_experiment_df, df, covariates=None):
    df = df.copy()
    X_train = pre_experiment_df[covariates]
    X_test = df[covariates]
    y_train = pre_experiment_df["Y"]
    gbm = HistGradientBoostingRegressor().fit(X_train, y_train)
    df["predictions"] = gbm.predict(X_test)
    return df

def get_ate_cupac(df, pre_experiment_df, covariates=None):
    df = add_cupac_predictions(pre_experiment_df, df, covariates=covariates)
    return get_ate_cuped(df, prediction_column="predictions")


def get_ate_doubly_robust(data, covariates, propensity_score: float = 0.5):
    data = data.copy().reset_index(drop=True)
    if not propensity_score:
        propensity_score = LogisticRegression(C=1e6, max_iter=1000).fit(data[covariates], data['T']).predict_proba(data[covariates])[:, 1]
    
    model_control = HistGradientBoostingRegressor()
    model_treatment = HistGradientBoostingRegressor()

    predictions_control = pd.Series(index=data.index, dtype=float)
    predictions_treatment = pd.Series(index=data.index, dtype=float)
    for fold in KFold(n_splits=2, shuffle=True, random_state=42).split(data):
        data_fold_0 = data.iloc[fold[0]]
        data_fold_1 = data.iloc[fold[1]]
        data_control_fold_0 = data_fold_0.query('T==0')
        data_treatment_fold_0 = data_fold_0.query('T==1')
        model_control.fit(
            data_control_fold_0[covariates], data_control_fold_0['Y']
        )
        model_treatment.fit(
            data_treatment_fold_0[covariates], data_treatment_fold_0['Y']
        )

        predictions_control[fold[1]] = model_control.predict(data_fold_1[covariates])
        predictions_treatment[fold[1]] = model_treatment.predict(data_fold_1[covariates])

    return (
        np.mean(data['T']*(data['Y'] - predictions_treatment)/propensity_score + predictions_treatment) -
        np.mean((1-data['T'])*(data['Y'] - predictions_control)/(1-propensity_score) + predictions_control)
    )
