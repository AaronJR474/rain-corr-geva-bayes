![Cover](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Data/grenada_caribbean_map.png)
# rain-corr-geva-bayes: Rainfall Correlation and Generalized Extreme Value Analysis using Bayesian Inference - Application to Grenada

This repository provides a reproducible workflow for analysing fragmented daily rainfall records in data-limited settings. It implements a site-specific correlation framework to diagnose the spatial structure and relative non-stationarity of daily rainfall, and uses Bayesian stationary and non-stationary spatial correlation models within ordinary kriging for rainfall imputation. For comparison, simpler deterministic imputation approaches are also evaluated, including nearest-neighbour distance (NND) and inverse-distance weighting (IDW), the latter implemented through a linear radial basis function. The imputed rainfall archive is then used within Bayesian generalized extreme value (GEV) and generalized Pareto (GPD) analyses to estimate rainfall extremes and develop multi-station intensity-duration-frequency (IDF) relationships. 

In addition to supporting the associated study for Grenada, the repository delivers the first island-wide generalized extreme value analysis of Grenadian rainfall currently assembled in this form, including spatially coherent return-level [maps](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Maps) and associated uncertainty estimates. Although developed for Grenada, the workflow is designed to be transferable to other data-limited regions where rainfall records are sparse, incomplete, and spatially uneven.

**All Bayesian Inference Markov Chain Monte Carlo (MCMC) analysis were conducted using the Python package [numpyro](https://num.pyro.ai/en/stable/).**

Interactive daily rainfall extreme maps for Grenada are available below:

1. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the 80–20 train-test spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_D80_D20_OCCKII.html)

2. [ICCK MM2 rainfall extreme maps (GEV and GPD) using the full-dataset spatial correlation model](https://aaronjr474.github.io/rain-corr-geva-bayes/Maps/GEVA_Dtot_OCCKII.html)

## How to cite

The raw daily rainfall observations used in this repository were obtained from the National Water and Sewerage Authority, Grenada (NAWASA) and the Grenada Airports Authority (GAA). The cleaned and curated rainfall archive compiled for this study is referred to as the **Grenada Daily Rainfall Database (GRD DRDB) v1.0**. If you use this dataset, please cite it as:

```bash

```

The analysis products generated in this repository for Grenada, including the imputed rainfall dataset, spatial correlation model summaries, extreme rainfall summaries, and [mapped]((https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Maps)) outputs, are provided in the [Ouputs](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs) directory. If you use the code, derived outputs, or workflow implemented in this repository, please cite the repository as:

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

## Utilizing the repository

The repository is organized around a set of self-contained Jupyter notebooks, each supported primarily by the helper modules [corr_imput_utilities.py](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/corr_imput_utilities.py) and [return_levels.py](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/return_levels.py). In the context of the Grenada application, the workflow is broken down into the following steps.

### Step 1: Site-specific correlation analysis, Bayesian stationary and non-stationary spatial correlation model development, and data imputation

There are two notebooks for this step. One uses the [80–20 train-test split](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/CORR_IMPUT_D80_D20.ipynb) for Bayesian spatial correlation model development, while the other uses the [full dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/CORR_IMPUT_Dtot.ipynb). In the case of the latter, the notebook also includes temporally non-stationary analysis in which spatial correlation models are developed on a per-month basis, motivated by the site-specific correlation structure which exhibited temporal non-stationarity across months. The non-stationary spatial correlation model (Model EElev) uses elevation as a secondary covariate and, when paired with ordinary kriging (OK), appears to provide the strongest performance relative to the deterministic and machine-learning-based methods considered.

Imputed datasets are provided for the stationary spatial correlation model (Model E) developed on the [80–20 train-test dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/GRN_RAINFALL_MODELE_FINAL_n6_thr25mm_var0p25.csv) and the [full dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/GRN_RAINFALL_MODELE_FINAL_n6_thr25mm_var0p25_full.csv), as well as for the non-stationary spatial correlation model (Model EElev) for the [80–20 train-test dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/GRN_RAINFALL_MODELElev_FINAL_n6_thr25mm_var0p25.csv) and the [full dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/GRN_RAINFALL_MODELElev_FINAL_n6_thr25mm_var0p25_full.csv).

![beforeafterimputation](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/before_After_imputation_grd.png)

### Step 2: Peaks-Over-Threshold (POT) threshold selection

The [Peaks-Over-Threshold (POT) Selection notebook](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/POTS_THR_SELEC.ipynb) provides the analysis used to select the threshold, \(u\), across all stations. It uses the [pyextremes Python package](https://georgebv.github.io/pyextremes/), which provides mean residual life, parameter stability, and return-level stability plots across [all stations](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs/POTS_PARAMS_STABILITY_PLOTS). The threshold is then selected by systematically reviewing these plots, balancing the statistical validity of the GPD approximation against the need to retain a sufficiently informative exceedance sample for Bayesian inference. The final thresholds and their associated uncertainty for all 28 stations can be found [here](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/pots_thresholds.csv).

### Step 3: Generalized extreme value analysis (GEVA) priors

The [priors notebook](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/PRIORS.ipynb) provides an overview of the development of the weakly informative priors used in the generalized extreme value analysis, including both the generalized extreme value (GEV) and generalized Pareto distribution (GPD) formulations. In essence, it uses the [pyextremes Python package](https://georgebv.github.io/pyextremes/) to analytically generate distributions of the [parameters](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs/PRIORS_POTS_BM) for each station via bootstrapping. These results were then used to inform the [final priors](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/PRIORS_POTS_BM/priors.xlsx) adopted in the Bayesian inference.

### Step 4: GEVA Bayesian inference

[The GEVA Bayesian inference notebook](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/GEVA_BAYESMCMC.ipynb) carries out the Bayesian fitting for the GEV and GPD variants using the outputs from Steps 1–3. [Trace plots and summary parameters](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs/BAYES_MCMC) are also provided as outputs to demonstrate the stability of the fit.

### Step 5: GEVA geostatistical analysis using VarioCorreKrigE

This step involves the computation of return levels for 5, 10, 25, 50, 75, and 100 years using the imputed data from Step 1 and the parameters from the GEVA Bayesian Inference in step 4. Notebooks for both the [80–20 train-test split](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/VarioCorreKrigE_GEVA_D80_D20.ipynb) and the [full dataset](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/VarioCorreKrigE_GEVA_Dtot.ipynb) under Model EElev are provided as they do give different geostatistical structures. These return levels are then analysed using several geostatistical techniques, including Ordinary Kriging, Ordinary Cokriging, Intrinsic Collocated Cokriging Markov Model I, and Intrinsic Collocated Cokriging Markov Model II. This involves the construction of variograms, cross-variograms, residual variograms, and a linear model of coregionalization. Using leave-one-out cross-validation (LOO-CV), this step determines which method provides the most robust estimates at unknown locations. Return-level summaries can be found [here under filenames ending in _summary.csv](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs). These are z-standardized, but their associated means and standard deviations are also provided for back-transformation.

### Step 6: GEVA map creation

GEVA maps are created using Ordinary Cokriging and Intrinsic Collocated Cokriging Markov Model II based on return levels derived from the [80–20 train-test Model EElev](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/VarioCorreKrigE_GEVA_MAPS_D80_D20.ipynb) and the [full dataset Model EElev](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/VarioCorreKrigE_GEVA_MAPS_Dtot.ipynb). The mapped values are provided [here under filenames ending in _summary.csv](https://github.com/AaronJR474/rain-corr-geva-bayes/tree/main/Outputs).

![gevgpd_mean](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/mean_gev_gpd_tr10_25_50.png)

**Figure 1.** Mean predicted daily rainfall extremes across Grenada for the GEV (top row) and GPD (bottom row) models at return periods of 10 years (left), 25 years (middle), and 50 years (right).

![gevgpd_std](https://github.com/AaronJR474/rain-corr-geva-bayes/blob/main/Outputs/sigma_gev_gpd_tr10_25_50.png)

**Figure 2.** Standard deviation of predicted daily rainfall extremes across Grenada for the GEV (top row) and GPD (bottom row) models at return periods of 10 years (left), 25 years (middle), and 50 years (right).

## References

> Bingham, E., Chen, J. P., Jankowiak, M., Obermeyer, F., Pradhan, N., Karaletsos, T., Singh, R., Szerlip, P. A., Horsfall, P., & Goodman, N. D. (2019). Pyro: Deep universal probabilistic programming. Journal of Machine Learning Research, 20(28), 1–6.

> Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. arXiv. https://arxiv.org/abs/1912.11554

> Rampersad, A. J. (2026). Variogram, Correlation and Kriging Estimation (VarioCorreKrigE) (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.19216626



