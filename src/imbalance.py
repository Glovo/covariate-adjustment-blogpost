# %% We want to check the estimator error of:
# 1. Imbalanced split, no covariates
# 2. Imbalanced split, covariates
# 3. Balanced split, no covariates
# 4. Balanced split, covariates

# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from tqdm import tqdm


# %%
def get_ate_ols(df, covariates=None):
    covariates = [] if not covariates else covariates
    T = "T +" if covariates else "T"
    result = smf.ols(f"Y ~ {T}" + "+".join(covariates), data=df).fit()
    return result.params["T"]


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate_data(effect):

    df = pd.DataFrame()
    N = 1000
    y_slope_x = 3
    df["X"] = np.random.normal(size=N)
    df["T"] = (sigmoid(np.random.normal(size=N)) > 0.5).astype(int)
    df["Y"] = np.random.normal(size=N) + y_slope_x * df["X"] + effect * df["T"]
    return df


ates = []
effect = 0.2
for _ in tqdm(range(1000)):
    df = generate_data(effect)
    ate_ols_x = get_ate_ols(df, covariates=["X"])
    ate_ols_vainilla = get_ate_ols(df)

    x_means = df.groupby("T").mean()["X"]
    imbalance = x_means.max() - x_means.min()
    ates.append(
        {
            "Using covariate X": ate_ols_x,
            "Not using covariates": ate_ols_vainilla,
            "Imbalance": imbalance,
        }
    )


# %%
df_ates = pd.DataFrame(ates).assign(
    ate_covariate_error=lambda x: np.abs(x["Using covariate X"] - effect),
    ate_error=lambda x: np.abs(x["Not using covariates"] - effect),
    imbalance_cut=lambda x: pd.qcut(x["Imbalance"], q=5),
)

long_df = df_ates.melt(id_vars="imbalance_cut").rename(
    columns={"variable": "Method", "value": "Error estimate"}
)
# %%
sns.displot(
    x="Error estimate",
    data=long_df.query("Method == 'ate_covariate_error'"),
    fill=True,
    kind="kde",
    hue="imbalance_cut",
    common_norm=False,
    alpha=0.4,
    linewidth=0,
    bw_adjust=1.2,
)
plt.show()

sns.displot(
    x="Error estimate",
    data=long_df.query("Method == 'ate_error'"),
    fill=True,
    kind="kde",
    hue="imbalance_cut",
    common_norm=False,
    alpha=0.4,
    linewidth=0,
    bw_adjust=1.2,
)
plt.show()


# %%
