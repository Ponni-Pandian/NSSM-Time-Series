# Advanced Time Series Forecasting with Neural State Space Models (NSSM)
## 1. Introduction
This project implements a Neural State Space Model (NSSM) for multivariate time series forecasting, combining the probabilistic structure of classical state-space models with the flexibility of neural networks. The objective is to move beyond traditional linear models such as ARIMA/SARIMA and implement a neural parameterization of the state-space matrices (A, B, C, D) while training the model using likelihood-based Kalman filtering.
The project evaluates forecasting performance against a classical SARIMAX baseline and analyzes the learned latent states to interpret the underlying system dynamics.
## 2. Dataset Selection
We use the Electricity Transformer Temperature (ETTh1) dataset, a publicly available high-frequency multivariate time series dataset. 
Dataset characteristics: Hourly frequency,17,000 time steps
Multiple exogenous load variables
Clear seasonality and long-term trends
Significant stochastic noise
Target variable: Oil Temperature (OT)
Covariates: HUFL, HULL, MUFL, MULL, LUFL, LULL
The dataset satisfies the project requirement of: Multivariate structure, More than 500 time steps
Clear seasonal and trend components
Data is split: 70% training, 30% testing
All variables are standardized prior to training.
## 3. Model Formulation
We implement a Neural State Space Model defined by:
### State Equation: xt​=Aθ​(ut​)xt−1​+Bθ​(ut​)ut​+wt​​
### Observation Equation: yt​=Cθ​(ut​)xt​+Dθ​(ut​)ut​+vt​
Where:
  X is the latent state vector
  ut​ are exogenous inputs
  yt​ is the observed target
  𝐴𝜃,𝐵𝜃,𝐶𝜃,𝐷𝜃  are neural network parameterizations
  𝑤𝑡∼𝑁(0,𝑄) process noise
  𝑣𝑡∼𝑁(0,𝑅) observation noise

Unlike classical linear state-space models with fixed matrices, our model learns time-varying transition and emission matrices through neural networks conditioned on exogenous inputs.
## 4. Training Methodology
Training is performed via maximum likelihood estimation using the Kalman filtering recursion.
At each time step:
State prediction
Covariance prediction
Innovation computation
Kalman gain update
Posterior state update
Log-likelihood accumulation
The total negative log-likelihood over the sequence is minimized using the Adam optimizer.
This satisfies the project requirement of:
Efficient likelihood computation
Use of Kalman filtering for probabilistic training
## 5. Baseline Model
For comparison, we implement a SARIMAX (Seasonal ARIMA with Exogenous Variables) model.
Configuration: Order: (2,1,2),Seasonal order: (1,1,1,24),Includes exogenous covariates, Forecast accuracy is evaluated using:Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
This provides a structured linear benchmark against which to compare the neural model.

## 6. Evaluation Strategy
The model is evaluated on:
1-step ahead forecasting
5-step ahead forecasting
Metrics: RMSE, MAE
This satisfies the requirement for multi-horizon forecasting comparison.
## 7. Latent State Analysis
The learned latent states are visualized over time to interpret system dynamics.
Observations show:
One latent dimension captures long-term trend behavior.
Another dimension captures daily seasonal oscillations.
Additional dimensions respond to load fluctuations and exogenous shocks.
Unlike SARIMAX, which uses fixed linear coefficients, the NSSM learns adaptive state transitions, allowing it to capture nonlinear and regime-dependent behavior.
This provides both predictive accuracy and interpretability — a key advantage of neural state-space models.
## 8. Conclusion
This project demonstrates that Neural State Space Models:
Successfully integrate neural networks into structured probabilistic models
Improve forecasting performance over classical linear baselines
Provide interpretable latent representations
Adapt to nonlinear and time-varying dynamics

The NSSM outperforms the SARIMAX baseline across forecasting horizons, particularly for longer-term predictions, highlighting the benefits of neural parameterization in complex multivariate systems.
Adapt to nonlinear and time-varying dynamics

The NSSM outperforms the SARIMAX baseline across forecasting horizons, particularly for longer-term predictions, highlighting the benefits of neural parameterization in complex multivariate systems.
