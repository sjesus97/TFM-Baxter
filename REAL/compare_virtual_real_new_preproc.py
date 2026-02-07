
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
FILE_VIRT = "data_train_phi75_theta145_butter_fc15.csv"
FILE_REAL = "data_train_phi75_theta145_art_def_butter_fc15.csv" 

OUT_DIR = "COMPARE_VIRTUAL_REAL_FC15"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_JOINTS = 7
DT = 1/500.0
MAX_LAG = 200
EPS = 1e-12
PLOT_SAMPLES = 4000  # para visualizar

# =========================
# AUX
# =========================
def rms(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x**2)))

def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def centered_xcorr_best_lag(x, y, max_lag=200):
    """
    Xcorr robusta: siempre recorta para que xs e ys tengan la misma longitud.
    Devuelve: best_lag, best_corr
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    N = min(len(x), len(y))
    x = x[:N] - np.mean(x[:N])
    y = y[:N] - np.mean(y[:N])

    denom = (np.linalg.norm(x) * np.linalg.norm(y) + EPS)

    best_corr = -1e9
    best_lag = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[-lag:N]
            ys = y[0:N + lag]
        else:
            xs = x[0:N - lag]
            ys = y[lag:N]

        L = min(len(xs), len(ys))
        if L < 50:
            continue

        xs2 = xs[:L]
        ys2 = ys[:L]
        c = float(np.dot(xs2, ys2) / denom)

        if c > best_corr:
            best_corr = c
            best_lag = lag

    return best_lag, best_corr

def align_by_lag(x_ref, y_real, lag):
    """
    
    lag > 0: y_real va retrasada -> recortar inicio de y_real
    lag < 0: y_real va adelantada -> recortar inicio de x_ref
    """
    x = np.asarray(x_ref, dtype=float)
    y = np.asarray(y_real, dtype=float)
    N = min(len(x), len(y))
    x = x[:N]
    y = y[:N]

    if lag > 0:
        y2 = y[lag:]
        x2 = x[:len(y2)]
    elif lag < 0:
        x2 = x[-lag:]
        y2 = y[:len(x2)]
    else:
        x2, y2 = x, y

    L = min(len(x2), len(y2))
    return x2[:L], y2[:L]

# =========================
# LOAD
# =========================
print(f"Leyendo virtual: {FILE_VIRT}")
df_v = pd.read_csv(FILE_VIRT)

print(f"Leyendo real   : {FILE_REAL}")
df_r = pd.read_csv(FILE_REAL)

cols_v = list(df_v.columns)
cols_r = list(df_r.columns)

print("\n=== Columnas ===")
print("Virtual:", cols_v[:10], "...", f"({len(cols_v)} cols)")
print("Real   :", cols_r[:10], "...", f"({len(cols_r)} cols)")

# Usar solo columnas comunes en el orden de virtual
common_cols = [c for c in cols_v if c in cols_r]
df_v = df_v[common_cols]
df_r = df_r[common_cols]

N = min(len(df_v), len(df_r))
df_v = df_v.iloc[:N].reset_index(drop=True)
df_r = df_r.iloc[:N].reset_index(drop=True)

print(f"\nN virtual={len(df_v)}, N real={len(df_r)} -> usamos N={N}")

signals = ["q", "dq", "ddq", "tau"]

# =========================
# LAG por joint con q
# =========================
print("\n================= LAG (XCORR) POR JOINT EN q =================")
lags = {}
for j in range(1, NUM_JOINTS+1):
    col = f"q{j}"
    if col not in df_v.columns:
        continue
    x = df_v[col].to_numpy()
    y = df_r[col].to_numpy()
    lag, c = centered_xcorr_best_lag(x, y, max_lag=MAX_LAG)
    lags[j] = lag
    print(f"Joint {j}: lag={lag} muestras ({lag*DT:.3f} s), xcorr={c:.3f}")

# =========================
# COMPARACIÓN NUMÉRICA (ALINEADA)
# =========================
print("\n================= COMPARACIÓN (ALINEADA POR LAG) =================")
rows = []
for j in range(1, NUM_JOINTS+1):
    lag = lags.get(j, 0)
    for sig in signals:
        col = f"{sig}{j}"
        if col not in df_v.columns:
            continue
        xv = df_v[col].to_numpy()
        yr = df_r[col].to_numpy()
        xv_a, yr_a = align_by_lag(xv, yr, lag)

        mean_v = float(np.mean(xv_a))
        mean_r = float(np.mean(yr_a))
        rms_v = rms(xv_a)
        rms_r = rms(yr_a)
        rmse = rms(yr_a - xv_a)
        corr = safe_corr(xv_a, yr_a)
        rel_rmse = 100.0 * rmse / (rms_v + EPS)

        rows.append({
            "joint": j, "sig": sig, "lag_samples": lag,
            "mean_v": mean_v, "mean_r": mean_r,
            "rms_v": rms_v, "rms_r": rms_r,
            "rmse": rmse, "rmse_rel_vs_v_%": rel_rmse,
            "corr": corr
        })

        print(f"J{j} {col}: corr={corr:.3f}, RMSE={rmse:.4f} ({rel_rmse:.1f}%), lag={lag}")

out_summary = os.path.join(OUT_DIR, "summary_aligned_metrics.csv")
pd.DataFrame(rows).to_csv(out_summary, index=False)
print(f"\n[OK] Métricas alineadas guardadas en: {out_summary}")

# =========================
# FIGURAS (ALINEADAS)
# =========================
print(f"\nGenerando figuras (alineadas) en: {OUT_DIR}")
for j in range(1, NUM_JOINTS+1):
    lag = lags.get(j, 0)
    for sig in signals:
        col = f"{sig}{j}"
        if col not in df_v.columns:
            continue
        xv = df_v[col].to_numpy()
        yr = df_r[col].to_numpy()
        xv_a, yr_a = align_by_lag(xv, yr, lag)

        L = min(PLOT_SAMPLES, len(xv_a))
        t = np.arange(L)

        plt.figure(figsize=(10,4))
        plt.plot(t, xv_a[:L], label=f"{col} virtual (aligned)")
        plt.plot(t, yr_a[:L], "--", label=f"{col} real (aligned)")
        plt.xlabel("muestra (aligned)")
        plt.ylabel(col)
        plt.title(f"{col} — virtual vs real (aligned), J{j}, lag={lag} samples")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        fname = os.path.join(OUT_DIR, f"{col}_J{j}_aligned.png")
        plt.savefig(fname, dpi=140)
        plt.close()

print("\n[OK] Comparación completada.")

