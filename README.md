## Introduction

The goal of this repo is to run some analysis to support the covariate adjustment blogpost from the Glovo Engineering blog. The blogpost describes some methods to reduce variance using covariates in order to estimate effects in an AB test. In this repo we have code to show, via simulations, the performance of such methods, organized as follows:
* Utilites in `src/` to run the desired simulations
* Notebooks with the actual simulations in `notebooks/`


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


### Contact

You can email:
* victor.bouzas@glovoapp.com
* david.masip@glovoapp.com
In case you have any doubts or suggestions to improve.
