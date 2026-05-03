![Cover](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Data/grenada_caribbean_map.png)
# rain-corr-geva-bayes: Rainfall Correlation and Generalized Extreme Value Analysis using Bayesian Inference - Application to Grenada

This repository provides a reproducible workflow for analysing fragmented daily rainfall records in data-limited settings. It implements a site-specific correlation framework to diagnose the spatial structure and relative non-stationarity of daily rainfall, and uses Bayesian stationary and non-stationary spatial correlation models within ordinary kriging for rainfall imputation. For comparison, simpler deterministic imputation approaches are also evaluated, including nearest-neighbour distance (NND) and inverse-distance weighting (IDW), the latter implemented through a linear radial basis function. The imputed rainfall archive is then used within Bayesian generalized extreme value (GEV) and generalized Pareto (GPD) analyses to estimate rainfall extremes and develop multi-station intensity-duration-frequency (IDF) relationships. 

In addition to supporting the associated study for Grenada, the repository delivers the first island-wide generalized extreme value analysis of Grenadian rainfall currently assembled in this form, including spatially coherent return-level [maps](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Maps) and associated uncertainty estimates. Although developed for Grenada, the workflow is designed to be transferable to other data-limited regions where rainfall records are sparse, incomplete, and spatially uneven.

Interactive daily rainfall extreme maps for Grenada are available below:

1. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the 80–20 train-test spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_D80_D20_OCCKII.html)

2. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the full-dataset spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_Dtot_OCCKII.html)

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
