# Forecast-informed adaptive reinforcement learning for 5G downlink throughput (Fused vs Reactive)

Code for a simulation study that asks whether short-horizon downlink (DL) throughput forecasts make a 5G optimization agent more adaptive and better able to generalize to traces and environments it has never seen. An ensemble forecaster is fused into the state of tabular Q-learning and Deep Q-Network (DQN) agents, which are compared with Reactive agents that omit the forecast. The real data are drive-test measurements from the University College Cork (UCC) 5G dataset (Raca et al., 2020).

All throughput gains are **simulated**. Action effects are expert-specified from RF engineering practice and are not calibrated against measured interventions.

## Repository contents

| File | What it does |
|---|---|
| `01_fused_forecasting_and_rl.ipynb` | Stage 1 and 2. Forecasting on 83 traces (Track A: global models with covariates, Track B: global models without covariates, Track C: per-trace local models), benchmark against Santos and Almeida (2025), the inverse-RMSE-weighted Track A ensemble, and single-trace tabular Q-learning with Fused agents (single-objective SO and multi-objective MO, 20 matched windows, 10 seeds). Exports `phase1_rl_exports/`. |
| `02_reactive_rl_and_comparison.ipynb` | Stage 2. Reactive SO/MO agents (same environments, rewards and hyperparameters, no forecast in the state) and the paired Fused-minus-Reactive comparison over the same 10 seeds. |
| `PART3_MULTITRACE_DQN_4FOLD_CV_SYNTHETIC.ipynb` | Stage 3. Multi-trace DQN with a three-step decision lead, trace-level four-fold cross-validation, held-out confirmation traces, dose-response analysis, 100 frozen synthetic changing environments, Experiment B (forecaster refit on each synthetic trace) and Experiment C (persistent-action testbed). Self-contained: it refits the forecaster itself. |
| `synth_gen.py`, `generator_config.json` | Generator for the 100 synthetic traces (fixed master seed 20261103). |
| `requirements.txt` | Package versions for the multi-trace stage (Python 3.14.6). |
| `requirements_single_trace.txt` | Package versions for the 83-trace forecasting and single-trace tabular stage (Python 3.13.9). |

## Data

Neither dataset is stored in this repository.

- **UCC 5G production dataset** (Raca et al., 2020): download it and place the per-trace CSV files in a folder named `5G-production-dataset/` next to the notebooks, or edit the path in the configuration cell.
- **Synthetic traces**: regenerate them with

  ```
  python synth_gen.py Synthetic_5G_100traces
  ```

  This writes 34 Good, 33 Bad and 33 Ugly traces (CSV, 1 Hz) plus a `manifest.csv` with every drawn parameter. The Part 3 notebook expects the folder `Synthetic_5G_100traces/` next to it, outside `5G-production-dataset/`. The generator is independent of the UCC data and of any model result, and was frozen before any model was run on its output.

## How to run

1. Create an environment and install the packages. Use `requirements_single_trace.txt` (Python 3.13.9) for the 83-trace forecasting and single-trace tabular stage, and `requirements.txt` (Python 3.14.6) for the multi-trace stage: `pip install -r <file>`.
2. Run `01_fused_forecasting_and_rl.ipynb`, then `02_reactive_rl_and_comparison.ipynb` (single-trace tabular stage).
3. Run `PART3_MULTITRACE_DQN_4FOLD_CV_SYNTHETIC.ipynb` top to bottom (multi-trace stage). All artifacts go to `<RUN_TAG>_artifacts/` (default `part3_cv4_run1_artifacts`). After a kernel restart keep the same `RUN_TAG` to resume; change it to start from scratch. The run trains 320 DQN agents (4 regimes × 2 objectives × 4 folds × 10 seeds) for the cross-validation, and again for the persistent-action testbed.

Track C (AutoARIMA with expanding-window refits) in notebook 01 is slow; it checkpoints to a local `.pkl` file and can be resumed. If LightGBM and PyTorch crash together at import, import `torch` before `lightgbm`.

## Design (see the paper for details)

- **Splits.** Each trace is split 80% development / 20% chronological holdout. Normalization is per-trace min–max fitted on the development portion only; holdout values are not clipped.
- **Traces.** 83 cleaned traces; 20 eligible traces (≤5% zero-DL samples) form the RL study; 21 held-out confirmation traces (5–40% zero DL, outages) are never used to fit the forecaster, the agents or any selection.
- **Single-trace stage.** Tabular Q-learning, in-sample on one trace with overlapping windows, so it does not measure generalization.
- **Multi-trace stage.** DQN with trace-level four-fold cross-validation (15 train / 5 test traces), seeds 42–51, γ = 0, action lead of 3 steps. Agents act on a forecast for the step at which the action lands. Regimes: Reactive (7 inputs), Fused (8), Reactive-8 (8 inputs, last observed DL in the forecast slot) and FusedShuffled (forecast shuffled in time). The pre-declared primary comparison is Fused versus Reactive-8.
- **Rewards.** SO is the normalized post-action DL throughput. MO is a weighted sum of DL, RSRP, RSRQ, CQI, RSSI and SNR (0.50, 0.15, 0.15, 0.08, 0.10, 0.02). No forecast-dependent or volume terms.
- **Action effects.** The DL gain is state-dependent, `mult × DL + 0.25 × rec × max(0, 1 − DL)`; the formulas and the MO KPI shifts are in Appendix A of the paper (Table A1) and in the environment cell of the Part 3 notebook.
- **Persistent-action testbed (Experiment C).** One setting stays active until replaced, a 10% switching dip applies at the step a new setting lands, γ = 0.9, with dynamic-programming reference policies.
- **Synthetic stage.** Frozen UCC-trained forecaster and frozen agents are applied once, without retraining. Experiment B is a post hoc, exploratory run in which the forecaster is refit on each synthetic trace's first 80%.

## Summary of results (simulation; see the paper)

| Test | Result |
|---|---|
| Unseen eligible traces (4-fold CV), Fused − Reactive-8 | SO +1.57 Mbps, MO +1.56 Mbps; both intervals above zero (SUPPORTED) |
| 21 confirmation traces | Fused above Reactive and above shuffled forecast in SO and MO (criteria met) |
| 100 synthetic traces, frozen forecaster | No Fused advantage (NOT SUPPORTED); the frozen forecaster has negative skill there |
| 100 synthetic traces, refit forecaster (post hoc) | Fused advantage, driven by the Good tier |
| Persistent actions with switching cost | Small and not statistically robust on real held-out traces |
| Adaptivity (oracle-action agreement, response to regime shifts) | Present, and identical for Reactive, Reactive-8 and Fused: it comes from the radio-state inputs, not the forecast |

Single-trace tabular results (10 seeds): no consistent Fused advantage (SO 24.883% vs 24.892%; MO 24.249% vs 24.315%), and a constant best action matched the immediate-reward oracle.

## Reference

Raca, D., Leahy, D., Sreenan, C. J., & Quinlan, J. J. (2020). Beyond throughput, the next generation: A 5G dataset with channel and context metrics. In *Proceedings of the 11th ACM Multimedia Systems Conference (MMSys '20)* (pp. 303–308). ACM.
