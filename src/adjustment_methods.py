import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import numpy as np

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
    if not propensity_score:
        propensity_score = LogisticRegression(C=1e6, max_iter=1000).fit(data[covariates], data['T']).predict_proba(data[covariates])[:, 1]
        
    outcome_regression_control = HistGradientBoostingRegressor().fit(data.query('T==0')[covariates], data.query('T==0')['Y']).predict(data[covariates])
    outcome_regression_treatment = HistGradientBoostingRegressor().fit(data.query('T==1')[covariates], data.query('T==1')['Y']).predict(data[covariates])
    return (
        np.mean(data['T']*(data['Y'] - outcome_regression_treatment)/propensity_score + outcome_regression_treatment) -
        np.mean((1-data['T'])*(data['Y'] - outcome_regression_control)/(1-propensity_score) + outcome_regression_control)
    )
