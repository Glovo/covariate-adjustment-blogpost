from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_samples(
    mu_metric: float,
    sigma_metric: float,
    epsilon: float,
    treatment_effect: float,
    size: int,
    cov_mu_eps: Optional[List[Tuple[float, float]]] = None,
    non_linear: Optional[List[bool]] = None,
    p_binomial: float = 0.5,
    random_epsilon: float = 1.0,
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate a dummy dataframe with the outcome (Y) for treatment (T=1) and control (T=0)
    and covariates that could impact the outcome.

    Assumptions:
    - The variables are assumed to be normally distributed
    - The covariates are additive on the outcome.
    - The treatment is randomly assigned between the individuals.
    - The treatment effect is a constant number and fixed between individuals
    - The metric during it is just the metric plus an error term (no trend)

    Inspired by:
        https://bytepawn.com/reducing-variance-in-ab-testing-with-cuped.html
        https://www.degeneratestate.org/posts/2018/Jan/04/reducing-the-variance-of-ab-test-using-prior-information/

    Args:
        mu_metric: mean of the outcome metric
        sigma_metric: variance of the outcome metric
        epsilon: scale of the error term
        treatment_effect: value of the treatment effect
        size: number of samples
        cov_mu_eps: list with mean and variance for the covariates
        non_linear: list with bool indicating if there is a non-linear covariate
        p_binomial: float indicating distribution of treatment/control
        random_epsilon: list with mean and variance for the covariates
        seed: random seed, if fixed
    """
    if seed:
        np.random.seed(seed)
    
    metric_before = np.random.normal(loc=mu_metric, scale=sigma_metric, size=size)

    error_term = np.random.normal(loc=0, scale=epsilon, size=size)

    treatment_assignment = np.random.binomial(n=1, p=p_binomial, size=size)

    metric_during = metric_before + error_term + treatment_assignment * treatment_effect

    if cov_mu_eps:
        covariates = np.zeros((len(cov_mu_eps), size))
        for i, mu_eps in enumerate(cov_mu_eps):
            covariates[i, :] = np.random.normal(mu_eps[0], mu_eps[1], size)
            if non_linear and non_linear[i]:
                metric_during = metric_during + covariates[i, :] ** 2
            else:
                metric_during = metric_during + covariates[i, :]
    else:
        covariates = []

    d = {
        "Y": metric_during,
        "T": treatment_assignment,
        "Y_before": metric_before,
    }
    d_cov = {f"X_{i+1}": c for i, c in enumerate(covariates)}
    r_cov = {"R_1": np.random.normal(loc=0, scale=random_epsilon, size=size)}
    return pd.DataFrame({**d, **d_cov, **r_cov})
