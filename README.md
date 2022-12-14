### Setup
To install the requirements run:

```bash
> poetry install
```

To run jupyter-lab on this new env, run while in the env:
```
> python -m ipykernel install --user --name=covariate-adjustment
> jupyter lab
```


### Notebooks

To run the simulations with the comparisons of the covariate adjustment methods 
run the notebook on `notebooks/covariate_adjustment.ipynb`. The other notebook 
`notebooks/good_bad_covariates.ipynb` compare the use of bad and good covariates with 
those methods and generate the DAGs of the blogpost.
