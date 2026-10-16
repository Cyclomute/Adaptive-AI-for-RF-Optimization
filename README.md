# Forecast-informed tabular Q-learning for 5G downlink throughput (Fused vs Reactive)

Code for a simulation study that combines an ensemble throughput forecaster with tabular Q-learning for 5G downlink (DL) throughput decisions, using drive-test measurements from the University College Cork 5G dataset (Raca et al., 2020).

## Notebooks (run in order)

1. `notebooks/01_fused_forecasting_and_rl.ipynb`
   - Forecasting on 83 traces (Track A: global models with covariates, Track B: global models without covariates, Track C: per-trace local models), benchmark comparison with Santos and Almeida (2025), and the inverse-RMSE-weighted Track A ensemble (selected by expanding-window CV inside the 80% development portion).
   - Single-trace RL: Fused single-objective (SO) and multi-objective (MO) tabular Q-learning, 20 matched evaluation windows, and a 10-seed robustness run for the Fused agents.
   - Exports `phase1_rl_exports/` for the Reactive notebook.
2. `notebooks/02_reactive_rl_and_comparison.ipynb`
   - Reactive SO/MO agents (same environments, rewards and hyperparameters, no forecast in the state) and the paired Fused-minus-Reactive comparison over the same 10 seeds.

## Data

The dataset is not included. Download the UCC 5G production dataset (Raca et al., 2020) and place the per-trace CSV files in a folder named `5G-production-dataset/` next to the notebooks (or edit the data path in the configuration cell).

## Design notes

- Normalisation is per trace min-max, fitted on the chronological development portion only; holdout values are not clipped.
- SO reward: post-action normalised DL throughput. MO reward: weighted sum of normalised DL, RSRP, RSRQ, CQI, RSSI and SNR (0.50, 0.15, 0.15, 0.08, 0.10, 0.02). There are no forecast-dependent or volume terms in either reward.
- Action effects are prescribed and not calibrated against measured interventions, and the next observation does not depend on the action.
- RL agents are trained and evaluated on the same selected trace and the 20 windows overlap, so the RL results are in-sample and do not measure generalisation to unseen traces.

## Environment

Python 3.13, see `requirements.txt`. Track C (AutoARIMA with expanding-window refits) is slow; it checkpoints to a local `.pkl` file and can be resumed.

## Reference

Raca, D., Leahy, D., Sreenan, C. J., and Quinlan, J. J. (2020). Beyond throughput, the next generation: a 5G dataset with channel and context metrics. MMSys '20.
