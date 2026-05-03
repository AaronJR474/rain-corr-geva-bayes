![Cover](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Data/grenada_caribbean_map.png)
# rain-corr-geva-bayes: Rainfall Correlation and Generalized Extreme Value Analysis using Bayesian Inference - Application to Grenada

This repository provides a reproducible workflow for analysing fragmented daily rainfall records in data-limited settings. It implements a site-specific correlation framework to diagnose the spatial structure and relative non-stationarity of daily rainfall, and uses Bayesian stationary and non-stationary spatial correlation models within ordinary kriging for rainfall imputation. For comparison, simpler deterministic imputation approaches are also evaluated, including nearest-neighbour distance (NND) and inverse-distance weighting (IDW), the latter implemented through a linear radial basis function. The imputed rainfall archive is then used within Bayesian generalized extreme value (GEV) and generalized Pareto (GPD) analyses to estimate rainfall extremes and develop multi-station intensity-duration-frequency (IDF) relationships. 

In addition to supporting the associated study for Grenada, the repository delivers the first island-wide generalized extreme value analysis of Grenadian rainfall currently assembled in this form, including spatially coherent return-level [maps](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Maps) and associated uncertainty estimates. Although developed for Grenada, the workflow is designed to be transferable to other data-limited regions where rainfall records are sparse, incomplete, and spatially uneven.

Interactive daily rainfall extreme maps for Grenada are available below:

1. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the 80–20 train-test spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_D80_D20_OCCKII.html)

2. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the full-dataset spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_Dtot_OCCKII.html)

## How to cite

The raw daily rainfall observations used in this repository were obtained from the National Water and Sewerage Authority, Grenada (NAWASA) and the Grenada Airports Authority (GAA). The cleaned and curated rainfall archive compiled for this study is referred to as the **Grenada Daily Rainfall Database (GRD DRDB) v1.0**. If you use this dataset, please cite it as:

```bash

```

The analysis products generated in this repository for Grenada, including the imputed rainfall dataset, spatial correlation model summaries, extreme rainfall summaries, and mapped outputs, are provided in the [Ouputs](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs) directory. If you use the code, derived outputs, or workflow implemented in this repository, please cite the repository as:

```bash

```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/AaronJR474/rain-corr-geva-bayes.git
   cd rain-corr-geva-bayes
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   ```bash
   pip install "git+https://github.com/AaronJR474/VarioCorreKrigE.git#egg=VarioCorreKrigE[bayesmcmc]"
   ```

## Utilizing the package

The package is broken down into Jupyter notebooks each of which are self-contained and only relies on the helpers: [corr_imput_utilities.py](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/corr_imput_utilities.py) and [return_levels.py](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/return_levels.py). 
