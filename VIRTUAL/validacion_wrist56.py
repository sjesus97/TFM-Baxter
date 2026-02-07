import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 0) CONFIG
# ===============================
CSV_VAL    = "wrist56_val_like_train.csv"
CSV_TRAIN  = "wrist56_train_basic.csv"   
DIR_TRAIN  = "out_wrist56_train"
OUT_DIR    = "out_wrist56_val_like_train"
os.makedirs(OUT_DIR, exist_ok=True)

THETA_FULL = os.path.join(DIR_TRAIN, "theta_full_wls.npy")
THETA_J567 = os.path.join(DIR_TRAIN, "theta_block_J567.npy")
THETA_J67  = os.path.join(DIR_TRAIN, "theta_block_J67.npy")

NUM_JOINTS = 7
NUM_PARAMS_PER_JOINT = 10  # 70

assert os.path.exists(CSV_VAL),    f"No encuentro {CSV_VAL}"
for p in [THETA_FULL, THETA_J567, THETA_J67]:
    assert os.path.exists(p), f"No encuentro {p}"

# ===============================
# 1) AUXILIARES
# ===============================
@jit(nopython=True)
def dotMat(inputVec):
    return np.array([
        [inputVec[0], -inputVec[1], inputVec[2], 0.0, 0.0, 0.0],
        [0.0, inputVec[0], 0.0, inputVec[1], inputVec[2], 0.0],
        [0.0, 0.0, inputVec[0], 0.0, inputVec[1], inputVec[2]],
    ])

@jit(nopython=True)
def crossMat(inputVec):
    return np.array([
        [0.0, -inputVec[2],  inputVec[1]],
        [inputVec[2], 0.0, -inputVec[0]],
        [-inputVec[1], inputVec[0], 0.0],
    ])

@jit(nopython=True)
def s(parameter, joint): return np.sin(parameter[joint])

@jit(nopython=True)
def c(parameter, joint): return np.cos(parameter[joint])

@jit(nopython=True)
def transMat(link, alphas, thetas):
    return np.array([
        [c(thetas, link), -s(thetas, link), 0.0],
        [s(thetas, link) * c(alphas, link),  c(thetas, link) * c(alphas, link), -s(alphas, link)],
        [s(thetas, link) * s(alphas, link),  c(thetas, link) * s(alphas, link),  c(alphas, link)],
    ])

@jit(nopython=True)
def newtonEulerEstimate(angles, velocities, accels):
    # MISMO setup que TRAIN
    half_pi = np.pi / 2
    alphas = np.array([0.0, -half_pi, half_pi, -half_pi, half_pi, -half_pi, half_pi])
    z = np.array([0.0, 0.0, 1.0])
    pArray = np.array([
        [0.0,      0.0,      0.0     ],
        [0.069,    0.0,      0.27035 ],
        [0.0,     -0.102,    0.0     ],
        [0.069,    0.0,      0.26242 ],
        [0.0,     -0.10359,  0.0     ],
        [0.01,     0.0,      0.2707  ],
        [0.0,     -0.115975, 0.0     ]
    ]).T

    thetas = angles.copy()
    thetas[1] += half_pi  # offset Baxter

    angularVels = np.zeros((3, 7))
    angularAccs = np.zeros((3, 7))
    linearAccs  = np.zeros((3, 7))

    w_j_j = np.zeros((6, 10, 7))
    TMatrices = np.zeros((6, 6, 7))
    grav = np.array([0.0, 0.0, -9.81])

    for link in range(7):
        rotMat = transMat(link, alphas, thetas)
        if link == 0:
            angularVels[:, 0] = velocities[0] * z
            angularAccs[:, 0] = accels[0] * z
            linearAccs[:, 0]  = np.array([0.0, 0.0, 0.0])
        else:
            ang_prev = rotMat.T @ angularVels[:, link-1]
            angularVels[:, link] = ang_prev + velocities[link] * z
            angularAccs[:, link] = rotMat.T @ angularAccs[:, link-1] \
                                 + np.cross(ang_prev, velocities[link]*z + accels[link]*z)
            linearAccs[:, link]  = rotMat.T @ ( linearAccs[:, link-1]
                                   + np.cross(angularAccs[:, link-1], pArray[:, link])
                                   + np.cross(angularVels[:, link-1],
                                              np.cross(angularVels[:, link-1], pArray[:, link])) )

        W_up = np.concatenate((
            (linearAccs[:, link] - grav).reshape(3, 1),
            crossMat(angularAccs[:, link]) + crossMat(angularVels[:, link]) @ crossMat(angularVels[:, link]),
            np.zeros((3, 6))
        ), axis=1)
        W_lo = np.concatenate((
            np.zeros((3, 1)),
            crossMat(grav - linearAccs[:, link]),
            dotMat(angularAccs[:, link]) + crossMat(angularVels[:, link]) @ dotMat(angularVels[:, link])
        ), axis=1)
        w_j_j[:, :, link] = np.concatenate((W_up, W_lo), axis=0)

        T_up = np.concatenate((rotMat, np.zeros((3, 3))), axis=1)
        T_lo = np.concatenate((crossMat(pArray[:, link]) @ rotMat, rotMat), axis=1)
        TMatrices[:, :, link] = np.concatenate((T_up, T_lo), axis=0)

    w11 = w_j_j[:, :, 0]; w22 = w_j_j[:, :, 1]; w33 = w_j_j[:, :, 2]
    w44 = w_j_j[:, :, 3]; w55 = w_j_j[:, :, 4]; w66 = w_j_j[:, :, 5]; w77 = w_j_j[:, :, 6]
    T2 = TMatrices[:, :, 1]; T3 = TMatrices[:, :, 2]; T4 = TMatrices[:, :, 3]
    T5 = TMatrices[:, :, 4]; T6 = TMatrices[:, :, 5]; T7 = TMatrices[:, :, 6]

    U77 = w77
    U66 = w66;  U67 = T7 @ w77
    U55 = w55;  U56 = T6 @ w66;                  U57 = T6 @ T7 @ w77
    U44 = w44;  U45 = T5 @ w55;                  U46 = T5 @ T6 @ w66;               U47 = T5 @ T6 @ T7 @ w77
    U33 = w33;  U34 = T4 @ w44;                  U35 = T4 @ T5 @ w55;               U36 = T4 @ T5 @ T6 @ w66;         U37 = T4 @ T5 @ T6 @ T7 @ w77
    U22 = w22;  U23 = T3 @ w33;                  U24 = T3 @ T4 @ w44;               U25 = T3 @ T4 @ T5 @ w55;         U26 = T3 @ T4 @ T5 @ T6 @ w66;  U27 = T3 @ T4 @ T5 @ T6 @ T7 @ w77
    U11 = w11;  U12 = T2 @ w22;                  U13 = T2 @ T3 @ w33;               U14 = T2 @ T3 @ T4 @ w44;         U15 = T2 @ T3 @ T4 @ T5 @ w55;  U16 = T2 @ T3 @ T4 @ T5 @ T6 @ w66;  U17 = T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ w77

    obs = np.zeros((7, 70))
    last = 5

    obs[0,   0: 10] = U11[last, :]
    obs[0,  10: 20] = U12[last, :]
    obs[0,  20: 30] = U13[last, :]
    obs[0,  30: 40] = U14[last, :]
    obs[0,  40: 50] = U15[last, :]
    obs[0,  50: 60] = U16[last, :]
    obs[0,  60: 70] = U17[last, :]

    obs[1,  10: 20] = U22[last, :]
    obs[1,  20: 30] = U23[last, :]
    obs[1,  30: 40] = U24[last, :]
    obs[1,  40: 50] = U25[last, :]
    obs[1,  50: 60] = U26[last, :]
    obs[1,  60: 70] = U27[last, :]

    obs[2,  20: 30] = U33[last, :]
    obs[2,  30: 40] = U34[last, :]
    obs[2,  40: 50] = U35[last, :]
    obs[2,  50: 60] = U36[last, :]
    obs[2,  60: 70] = U37[last, :]

    obs[3,  30: 40] = U44[last, :]
    obs[3,  40: 50] = U45[last, :]
    obs[3,  50: 60] = U46[last, :]
    obs[3,  60: 70] = U47[last, :]

    obs[4,  40: 50] = U55[last, :]
    obs[4,  50: 60] = U56[last, :]
    obs[4,  60: 70] = U57[last, :]

    obs[5,  50: 60] = U66[last, :]
    obs[5,  60: 70] = U67[last, :]

    obs[6,  60: 70] = U77[last, :]

    return obs  # (7,70)

def row_idx_for_joint(j, N):
    return np.arange(j, N*NUM_JOINTS, NUM_JOINTS)

def rms(x): return float(np.sqrt(np.mean(np.square(x))))

def plot_q_dq_tau(t, q, dq, tau, tau_hat, j_idx, title, out_png):
    plt.figure(figsize=(12,6.5))
    ax1 = plt.subplot(3,1,1); ax1.plot(t, q[:, j_idx]); ax1.set_ylabel(f"q{j_idx+1} [rad]")
    ax1.grid(True); ax1.set_title(title)
    ax2 = plt.subplot(3,1,2); ax2.plot(t, dq[:, j_idx]); ax2.set_ylabel(f"dq{j_idx+1} [rad/s]"); ax2.grid(True)
    ax3 = plt.subplot(3,1,3); ax3.plot(t, tau[:, j_idx], label="τ real"); ax3.plot(t, tau_hat[:, j_idx], "--", label="τ̂")
    ax3.set_xlabel("muestra"); ax3.set_ylabel(f"τ{j_idx+1} [Nm]"); ax3.grid(True); ax3.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()

# ===============================
# 2) CARGA Y CONSTRUCCIÓN DE Y
# ===============================
def build_Y_tau_from_csv(csv_path):
    print(f"Leyendo: {csv_path}")
    df = pd.read_csv(csv_path)
    q   = df[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
    dq  = df[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
    ddq = df[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
    tau = df[[f"tau{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
    N = q.shape[0]
    print("Muestras:", N)

    Y = np.zeros((N*NUM_JOINTS, NUM_JOINTS*NUM_PARAMS_PER_JOINT))
    b = np.zeros(N*NUM_JOINTS)

    for i in range(N):
        obs = newtonEulerEstimate(q[i], dq[i], ddq[i])  # (7,70)
        base = i*NUM_JOINTS
        for j in range(NUM_JOINTS):
            Y[base + j, :] = obs[j, :]
            b[base + j]   = tau[i, j]
    return q, dq, ddq, tau, Y, b, N

# ===============================
# 3)Comparativa de columnas Y_val vs Y_train
# ===============================
def col_rms_from_csv(csv_path, stride=5, max_samples=None):
    """Acumula energía por columna de Y de forma 'streaming' (aprox, para diagnóstico rápido)."""
    if not os.path.exists(csv_path):
        return None
    print(f"[DIAG] Calculando col_rms(Y) aprox desde {csv_path} (stride={stride}, max={max_samples})")
    df_iter = pd.read_csv(csv_path, chunksize=2000)
    sumsq = np.zeros(70, dtype=np.float64)
    n_rows = 0
    total_samples = 0
    for chunk in df_iter:
        q   = chunk[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
        dq  = chunk[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
        ddq = chunk[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
        # stride
        idx = np.arange(0, q.shape[0], stride)
        for i in idx:
            obs = newtonEulerEstimate(q[i], dq[i], ddq[i])  # (7,70)
            # apila 7 filas -> energía por columna
            # rms por columna proporcional a raíz de media; aquí acumulamos suma de cuadrados
            sumsq += np.sum(obs**2, axis=0)
            n_rows += obs.shape[0]  # +7
            total_samples += 1
            if max_samples is not None and total_samples >= max_samples:
                break
        if max_samples is not None and total_samples >= max_samples:
            break
    if n_rows == 0:
        return None
    rms_cols = np.sqrt(sumsq / n_rows)
    return rms_cols

# ===============================
# 4) VALIDACIÓN
# ===============================
def main():
    # Y y datos de VALIDACIÓN
    q, dq, ddq, tau, Y, b, N = build_Y_tau_from_csv(CSV_VAL)

    # Carga thetas
    theta_full = np.load(THETA_FULL).ravel()
    theta_J567 = np.load(THETA_J567).ravel()
    theta_J67  = np.load(THETA_J67 ).ravel()

    # Predicción (las thetas ya están acolchadas a 70)
    tau_est_full  = (Y @ theta_full ).reshape(N, NUM_JOINTS)
    tau_est_567   = (Y @ theta_J567).reshape(N, NUM_JOINTS)
    tau_est_67    = (Y @ theta_J67 ).reshape(N, NUM_JOINTS)

    # Reporte J5/J6
    def report(name, tau_hat):
        print(f"\n=== Métricas por joint — {name} ===")
        for j in [4,5]:  # J5, J6
            mse = mean_squared_error(tau[:, j], tau_hat[:, j])
            r2  = r2_score(tau[:, j], tau_hat[:, j])
            print(f"J{j+1}: MSE={mse:.6f}  R2={r2:.4f}  RMS(τ)={rms(tau[:,j]):.4f}")

        print("\nMSE ponderado por joint (WLS, diagnóstico):")
        sigma = np.array([rms(tau[:, jj]) + 1e-12 for jj in range(NUM_JOINTS)], dtype=float)
        for j in [4,5]:
            err_j = (tau_hat[:, j] - tau[:, j]) / sigma[j]
            mse_w_j = float(np.mean(err_j**2))
            print(f"  J{j+1}: {mse_w_j:.6f}")

    report("FULL_WLS",     tau_est_full)
    report("BLOCK_J5J6J7", tau_est_567)
    report("BLOCK_J6J7",   tau_est_67)

    # Energía (RMS) VALIDACIÓN
    print("\n=== Energía (RMS) — señales articulares (VALIDACIÓN) ===")
    for j in range(NUM_JOINTS):
        print(f"J{j+1}: RMS(dq)={rms(dq[:,j]):.5f}  RMS(ddq)={rms(ddq[:,j]):.5f}  RMS(τ)={rms(tau[:,j]):.5f}")

    # Gráficas
    t = np.arange(N)
    combos = [("FULL_WLS", tau_est_full), ("BLOCK_J5J6J7", tau_est_567), ("BLOCK_J6J7", tau_est_67)]
    for name, tau_hat in combos:
        for j_idx in [4,5]:
            png = os.path.join(OUT_DIR, f"{name}_J{j_idx+1}.png")
            plot_q_dq_tau(t, q, dq, tau, tau_hat, j_idx, title=f"{name} — Joint {j_idx+1}", out_png=png)

    # Comparativa de columnas Y_val vs Y_train
    if os.path.exists(CSV_TRAIN):
        rms_val   = col_rms_from_csv(CSV_VAL,  stride=5, max_samples=8000)
        rms_train = col_rms_from_csv(CSV_TRAIN, stride=20, max_samples=16000)
        if rms_val is not None and rms_train is not None:
            ratio = np.divide(rms_val, np.maximum(rms_train, 1e-12))
            med = float(np.median(ratio)); rmin = float(np.min(ratio)); rmax = float(np.max(ratio))
            print(f"\n[DIAG] col_rms_val/col_rms_train — mediana={med:.3f}  [min={rmin:.3f}, max={rmax:.3f}]")
            with open(os.path.join(OUT_DIR, "col_rms_ratio.json"), "w") as f:
                json.dump(dict(median=med, min=rmin, max=rmax), f, indent=2)
        else:
            print("\n[DIAG] No se pudo calcular col_rms (csv inexistente o sin filas).")

    # Resumen
    summary = dict(
        csv_val=CSV_VAL,
        thetas=dict(full=THETA_FULL, J567=THETA_J567, J67=THETA_J67),
        N=int(N),
    )
    with open(os.path.join(OUT_DIR, "summary_val.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Resultados y gráficas en: {OUT_DIR}")

if __name__ == "__main__":
    main()

