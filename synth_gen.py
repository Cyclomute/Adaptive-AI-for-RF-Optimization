"""
Synthetic 5G downlink traces (good / bad / ugly) for out-of-sample testing.

Design principles
- Physics-based and independent of the UCC data: nothing here is fitted to UCC or to any forecasting/RL result.
- Covariates (RSRP, RSRQ, SNR, CQI, RSSI, Speed) are generated from latent radio and load processes;
  DL throughput is generated from the same latent state (truncated-Shannon spectral efficiency x bandwidth x
  per-user resource share), so the relationship covariates -> DL is realistic but NOT a deterministic function:
  the cell-load process is only partly visible through RSRQ/RSSI.
- Output matches the loader of the Part 3 notebook: Timestamp (%Y.%m.%d_%H.%M.%S), DL_bitrate (kbps),
  RSRP, RSRQ, SNR, CQI, RSSI, Speed; one CSV per trace under <root>/Synthetic_<Tier>/<Driving|Static>/.
- Fixed master seed; every trace stores its own seed and drawn parameters in manifest.csv.
- Zero-throughput share is capped below 3 % by construction and verified after generation.

Run:  python synth_gen.py <output_root>
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

MASTER_SEED = 20261103
N_PER_TIER = {"Good": 34, "Bad": 33, "Ugly": 33}
P_DRIVING = 0.6
BW_MHZ = 100.0
N_RB = 273
P_WIDE = {"Good": 0.30, "Bad": 0.0, "Ugly": 0.10}   # probability of a 200 MHz aggregate carrier (n78 + CA)
DL_CAP_MBPS = 500.0
ZERO_CAP = 0.029          # hard cap on exact-zero DL samples per trace (< 3 %)


def ar1(n, phi, sigma, rng):
    x = np.empty(n); x[0] = rng.normal(0, sigma)
    s = np.sqrt(1 - phi ** 2) * sigma
    e = rng.normal(0, s, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def sigmoid(x): return 1 / (1 + np.exp(-x))


def piecewise_offsets(n, rate_per_s, sigma, rng):
    """Piecewise-constant offset that jumps at Poisson event times (handovers / cell-edge regime switches)."""
    off = np.zeros(n); ev = []
    t = 0
    while True:
        t += rng.exponential(1.0 / rate_per_s) if rate_per_s > 0 else n + 1
        if t >= n: break
        ev.append(int(t))
    cur = 0.0; last = 0
    for e in ev:
        off[last:e] = cur; cur = rng.normal(0, sigma); last = e
    off[last:] = cur
    return off, ev


def speed_profile(n, driving, rng):
    if not driving:
        return np.abs(rng.normal(0, 0.3, n))
    tgt = np.zeros(n); i = 0
    while i < n:
        d = int(rng.integers(20, 90))
        v = 0.0 if rng.random() < 0.3 else rng.uniform(8, 55)
        tgt[i:i + d] = v; i += d
    v = np.empty(n); v[0] = tgt[0]
    for t in range(1, n):
        v[t] = v[t - 1] + 0.15 * (tgt[t] - v[t - 1])
    return np.clip(v + rng.normal(0, 0.5, n), 0, None)


TIERS = {
    # parameter ranges (uniform draws) per tier
    "Good": dict(rsrp_mu=(-92, -80), intf=(14, 20), sh_sig=(1.5, 3.0), fast_sig=(0.5, 1.2), ho_rate=(0.0, 0.002), ho_sig=(2, 4),
                 load_mu=(0.25, 0.45), load_sig=(0.2, 0.5), load_phi=(0.985, 0.995), alloc=(0.12, 0.32),
                 fades=(0, 0), app_limited=0.0, zero_budget=(0.000, 0.006)),
    "Bad":  dict(rsrp_mu=(-106, -96), intf=(16, 22), sh_sig=(3.5, 5.5), fast_sig=(1.5, 2.5), ho_rate=(0.002, 0.006), ho_sig=(3, 6),
                 load_mu=(0.55, 0.80), load_sig=(0.3, 0.7), load_phi=(0.97, 0.99), alloc=(0.08, 0.20),
                 fades=(0, 2), app_limited=0.0, zero_budget=(0.002, 0.015)),
    "Ugly": dict(rsrp_mu=(-98, -82), intf=(16, 30), sh_sig=(4.0, 6.5), fast_sig=(2.0, 3.5), ho_rate=(0.006, 0.02), ho_sig=(7, 12),
                 load_mu=(0.30, 0.75), load_sig=(0.8, 1.6), load_phi=(0.93, 0.98), alloc=(0.06, 0.22),
                 fades=(3, 9), app_limited=0.45, zero_budget=(0.010, 0.029)),
}


def draw(rng, lo_hi): return float(rng.uniform(*lo_hi))


def make_trace(tier, driving, seed):
    rng = np.random.default_rng(seed)
    P = TIERS[tier]
    n = int(rng.integers(600, 2001))
    prm = {k: (draw(rng, v) if isinstance(v, tuple) and len(v) == 2 and k not in ("fades",) else v) for k, v in P.items()}
    spd = speed_profile(n, driving, rng)
    bw = 200.0 if rng.random() < P_WIDE[tier] else BW_MHZ
    n_rb = int(N_RB * bw / BW_MHZ)

    # --- radio state ---
    shadow = ar1(n, 0.985, prm["sh_sig"], rng)
    fast = ar1(n, 0.55, prm["fast_sig"], rng) * (1 + spd / 40.0)
    off, ho_events = piecewise_offsets(n, prm["ho_rate"] if driving else prm["ho_rate"] * 0.2, prm["ho_sig"], rng)
    fade = np.zeros(n)
    for _ in range(int(rng.integers(P["fades"][0], P["fades"][1] + 1))):
        c = int(rng.integers(0, n)); w = int(rng.integers(3, 13)); depth = rng.uniform(12, 25)
        idx = np.arange(max(0, c - w), min(n, c + w + 1))
        fade[idx] -= depth * np.hanning(len(idx) + 2)[1:-1]
    rsrp = np.clip(prm["rsrp_mu"] + shadow + fast + off + fade + rng.normal(0, 0.4, n), -140, -44)

    # --- interference rise (slow) and cell load (latent, logit-AR) ---
    intf = np.clip(prm["intf"] + ar1(n, 0.98, 2.0, rng), 4, 40)
    base_logit = np.log(prm["load_mu"] / (1 - prm["load_mu"]))
    z = ar1(n, prm["load_phi"], prm["load_sig"], rng)
    # load bursts: abrupt level shifts (more frequent in ugly)
    burst_off, _ = piecewise_offsets(n, {"Good": 0.0005, "Bad": 0.002, "Ugly": 0.01}[tier], {"Good": 0.3, "Bad": 0.6, "Ugly": 1.3}[tier], rng)
    load = sigmoid(base_logit + z + burst_off)

    sinr = np.clip(rsrp + 122.0 - intf, -10, 30)                         # per-RE noise ~ -122 dBm (30 kHz, 7 dB NF)
    rsrq = np.clip(-3.0 - 7.8 * load - 10 * np.log10(1 + 10 ** (-sinr / 10)) + rng.normal(0, 0.3, n), -20, -3)
    rssi = np.clip(rsrp + 10 * np.log10(n_rb) - rsrq + rng.normal(0, 0.5, n), -110, -40)
    sinr_rep = np.roll(sinr, 1); sinr_rep[0] = sinr[0]                    # reporting delay 1 s
    snr_meas = np.clip(sinr_rep + rng.normal(0, 0.5, n), -10, 30)
    cqi = np.clip(np.round(1 + 0.524 * (sinr_rep + 6.7) + rng.normal(0, 0.6, n)), 1, 15)
    rsrp_meas = np.clip(rsrp + rng.normal(0, 0.5, n), -140, -44)

    # --- downlink throughput from the same latent state ---
    se = np.clip(0.75 * np.log2(1 + 10 ** (sinr / 10)), 0, 7.4)           # truncated Shannon, implementation loss
    bler = np.clip(0.1 + 0.002 * spd, 0.1, 0.25)
    share = (1 - load) * prm["alloc"]
    layers = np.clip(1 + (sinr - 8) / 9.0, 1, 4)                          # spatial layers (MIMO rank) grow with SINR
    inst = bw * se * layers * share * (1 - bler) * (1 - 0.003 * np.clip(spd, 0, 60))
    inst *= np.exp(rng.normal(0, 0.12, n))                                # scheduler / TCP noise
    knee = 350.0                                                          # soft ceiling: no flat plateau at the cap
    inst = np.where(inst > knee, knee + (DL_CAP_MBPS - knee) * np.tanh((inst - knee) / (DL_CAP_MBPS - knee)), inst)
    # application-limited segments (video-like)
    if rng.random() < P["app_limited"]:
        i = 0
        while i < n:
            d = int(rng.integers(10, 41)); cap = rng.uniform(5, 60)
            inst[i:i + d] = np.minimum(inst[i:i + d], cap); i += d
    # handover interruptions (throughput collapse for 1-3 s, not necessarily zero)
    for e in ho_events:
        d = int(rng.integers(1, 4)); inst[e:e + d] *= rng.uniform(0.0, 0.25)
    # application-level smoothing -> realistic autocorrelation
    dl = np.empty(n); dl[0] = inst[0]
    for t in range(1, n):
        dl[t] = 0.5 * dl[t - 1] + 0.5 * inst[t]
    dl = np.clip(dl, 0.01, None)

    # outages (exact zeros), total strictly below the 3 % cap
    budget = int(min(prm["zero_budget"], ZERO_CAP) * n)
    used = 0
    while used < budget:
        d = int(min(rng.integers(1, 9 if tier == "Ugly" else 4), budget - used))
        s = int(rng.integers(0, n - d)); dl[s:s + d] = 0.0; used += d
    zero_frac = float((dl == 0).mean())
    assert zero_frac < 0.03, zero_frac

    start = pd.Timestamp("2026-01-05 08:00:00") + pd.Timedelta(seconds=int(rng.integers(0, 60 * 86400)))
    ts = pd.date_range(start, periods=n, freq="1s").strftime("%Y.%m.%d_%H.%M.%S")
    df = pd.DataFrame({"Timestamp": ts, "DL_bitrate": np.round(dl * 1000.0, 1),
                       "RSRP": np.round(rsrp_meas, 1), "RSRQ": np.round(rsrq, 1), "SNR": np.round(snr_meas, 1),
                       "CQI": cqi.astype(int), "RSSI": np.round(rssi, 1), "Speed": np.round(spd, 2)})
    info = {"tier": tier, "mobility": "Driving" if driving else "Static", "rows": n, "bw_mhz": bw, "seed": seed, "zero_frac": round(zero_frac, 5),
            "n_handovers": len(ho_events), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in prm.items()}}
    return df, info


def main(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(MASTER_SEED); rows = []; k = 0
    for tier, cnt in N_PER_TIER.items():
        for j in range(cnt):
            k += 1; driving = bool(master.random() < P_DRIVING); seed = int(master.integers(1, 2 ** 31 - 1))
            df, info = make_trace(tier, driving, seed)
            folder = root / f"Synthetic_{tier}" / info["mobility"]; folder.mkdir(parents=True, exist_ok=True)
            name = f"S{k:03d}_{tier}_{info['mobility']}.csv"; df.to_csv(folder / name, index=False)
            info["file"] = str((folder / name).relative_to(root)); rows.append(info)
    man = pd.DataFrame(rows); man.to_csv(root / "manifest.csv", index=False)
    json.dump({"master_seed": MASTER_SEED, "n_per_tier": N_PER_TIER, "p_driving": P_DRIVING, "zero_cap": ZERO_CAP,
               "bw_mhz": BW_MHZ, "n_rb": N_RB}, open(root / "generator_config.json", "w"), indent=2)
    print("traces:", len(man), "| max zero fraction:", man.zero_frac.max())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Synthetic_5G_100traces")
