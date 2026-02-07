
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 0) CONFIG
# ===============================
CSV_INPUT = "wrist56_train_basic.csv"
SAVE_DIR  = "out_wrist56_train"
os.makedirs(SAVE_DIR, exist_ok=True)

SAVE_THETA_FULL   = os.path.join(SAVE_DIR, "theta_full_wls.npy")
SAVE_THETA_J567   = os.path.join(SAVE_DIR, "theta_block_J567.npy")  # (70,)
SAVE_THETA_J67    = os.path.join(SAVE_DIR, "theta_block_J67.npy")   # (70,)
SAVE_THETA_J7     = os.path.join(SAVE_DIR, "theta_block_J7.npy")    # (70,)
SAVE_WSIGMA       = os.path.join(SAVE_DIR, "w_sigma.npy")
PLOTS_DIR         = os.path.join(SAVE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

NUM_JOINTS = 7
NUM_PARAMS_PER_JOINT = 10   # 7x10 = 70


LAST_IDX = 5 

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

# ===============================
# 2) NEWTON–EULER (7,70)
# ===============================
@jit(nopython=True)
def newtonEulerEstimate(angles, velocities, accels, last_idx=5):
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
    thetas[1] += half_pi

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
    li = last_idx 

    obs[0,   0: 10] = U11[li, :]
    obs[0,  10: 20] = U12[li, :]
    obs[0,  20: 30] = U13[li, :]
    obs[0,  30: 40] = U14[li, :]
    obs[0,  40: 50] = U15[li, :]
    obs[0,  50: 60] = U16[li, :]
    obs[0,  60: 70] = U17[li, :]

    obs[1,  10: 20] = U22[li, :]
    obs[1,  20: 30] = U23[li, :]
    obs[1,  30: 40] = U24[li, :]
    obs[1,  40: 50] = U25[li, :]
    obs[1,  50: 60] = U26[li, :]
    obs[1,  60: 70] = U27[li, :]

    obs[2,  20: 30] = U33[li, :]
    obs[2,  30: 40] = U34[li, :]
    obs[2,  40: 50] = U35[li, :]
    obs[2,  50: 60] = U36[li, :]
    obs[2,  60: 70] = U37[li, :]

    obs[3,  30: 40] = U44[li, :]
    obs[3,  40: 50] = U45[li, :]
    obs[3,  50: 60] = U46[li, :]
    obs[3,  60: 70] = U47[li, :]

    obs[4,  40: 50] = U55[li, :]
    obs[4,  50: 60] = U56[li, :]
    obs[4,  60: 70] = U57[li, :]

    obs[5,  50: 60] = U66[li, :]
    obs[5,  60: 70] = U67[li, :]

    obs[6,  60: 70] = U77[li, :]

    return obs  # (7,70)

# ============== utilidades ==============
def row_idx_for_joint(j, N, n_joints=7):
    return np.arange(j, N*n_joints, n_joints)

def rms(x): return float(np.sqrt(np.mean(x**2)))

# ===============================
# 3) CARGA
# ===============================
print(f"Leyendo {CSV_INPUT} ...")
df = pd.read_csv(CSV_INPUT)
q   = df[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
dq  = df[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
ddq = df[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
tau = df[[f"tau{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
N = q.shape[0]
print(f"Muestras: {N}")

# ===============================
# 4) Y (N*7 x 70) y τ_vec (N*7,)
# ===============================
matObs70 = np.zeros((N*NUM_JOINTS, NUM_JOINTS*NUM_PARAMS_PER_JOINT))
tau_vec  = np.zeros(N*NUM_JOINTS)

for i in range(N):
    obs = newtonEulerEstimate(q[i], dq[i], ddq[i], last_idx=LAST_IDX)  # (7,70)
    base = i*NUM_JOINTS
    for j in range(NUM_JOINTS):
        matObs70[base + j, :] = obs[j, :]
        tau_vec[base + j]     = tau[i, j]

print("matObs70:", matObs70.shape, "tau_vec:", tau_vec.shape)

# ===============================
# 5) WLS global (por junta) + normalización de columnas
# ===============================
sigma = np.zeros(NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = row_idx_for_joint(j, N, NUM_JOINTS)
    sigma[j] = rms(tau_vec[idx]) + 1e-12
np.save(SAVE_WSIGMA, sigma)
print("Escalas WLS (RMS por joint):", np.round(sigma, 6))

w = np.zeros(N*NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = row_idx_for_joint(j, N, NUM_JOINTS)
    w[idx] = 1.0 / sigma[j]
Wsqrt = np.sqrt(w)
A_full = matObs70 * Wsqrt[:, None]
b_full = tau_vec * Wsqrt


col_rms_full = np.sqrt(np.mean(A_full**2, axis=0)) + 1e-12
A_full_n = A_full / col_rms_full[None, :]
theta_full_n, *_ = np.linalg.lstsq(A_full_n, b_full, rcond=None)
theta_full = theta_full_n / col_rms_full
np.save(SAVE_THETA_FULL, theta_full)
print(f"[OK] theta_full guardado en {SAVE_THETA_FULL}")

# ===============================
# 6) BLOQUES MUÑECA
# ===============================
COLS_J7   = slice(60, 70)   # 10
COLS_J67  = slice(50, 70)   # 20
COLS_J567 = slice(40, 70)   # 30

ROWS_J7   = row_idx_for_joint(6, N, NUM_JOINTS)
ROWS_J6   = row_idx_for_joint(5, N, NUM_JOINTS)
ROWS_J5   = row_idx_for_joint(4, N, NUM_JOINTS)
ROWS_J67  = np.sort(np.concatenate([ROWS_J6, ROWS_J7]))
ROWS_J567 = np.sort(np.concatenate([ROWS_J5, ROWS_J6, ROWS_J7]))

def fit_block_padded(rows, cols):
    """Resuelve el bloque y devuelve θ acolchado (70,), más rank, svals, cond (sin pesos)."""
    A = matObs70[rows, cols]
    b = tau_vec[rows]

    
    sigma_row = np.empty(rows.shape[0])
    for k, r in enumerate(rows):
        sigma_row[k] = sigma[r % NUM_JOINTS]
    Wsqrt_loc = 1.0 / np.sqrt(sigma_row + 1e-12)
    A_w = A * Wsqrt_loc[:, None]
    b_w = b * Wsqrt_loc

    
    col_rms = np.sqrt(np.mean(A_w**2, axis=0)) + 1e-12
    A_wn = A_w / col_rms[None, :]

    theta_local_n, residuals, rank, svals = np.linalg.lstsq(A_wn, b_w, rcond=None)
    theta_local = theta_local_n / col_rms

    
    theta_pad = np.zeros(70)
    theta_pad[cols] = theta_local

    
    try:
        S = np.linalg.svd(A, full_matrices=False)[1]
        cond = (S.max() / S.min()) if np.all(S > 1e-12) else np.inf
    except Exception:
        S, cond = np.array([]), np.nan

    return theta_pad, rank, svals, cond

theta_J7,   rank_J7,   s_J7,   cond_J7   = fit_block_padded(ROWS_J7,   COLS_J7)
theta_J67,  rank_J67,  s_J67,  cond_J67  = fit_block_padded(ROWS_J67,  COLS_J67)
theta_J567, rank_J567, s_J567, cond_J567 = fit_block_padded(ROWS_J567, COLS_J567)

np.save(SAVE_THETA_J7,   theta_J7)
np.save(SAVE_THETA_J67,  theta_J67)
np.save(SAVE_THETA_J567, theta_J567)
print(f"[OK] θ_J7→{SAVE_THETA_J7}     rank={rank_J7}   cond={cond_J7:.3e}")
print(f"[OK] θ_J67→{SAVE_THETA_J67}   rank={rank_J67}  cond={cond_J67:.3e}")
print(f"[OK] θ_J567→{SAVE_THETA_J567} rank={rank_J567} cond={cond_J567:.3e}")

# ===============================
# 7) MÉTRICAS Y GRÁFICAS (q, dq, τ vs τ̂)
# ===============================
def eval_and_plot(theta_vec, tag):
    tau_est_vec = matObs70 @ theta_vec
    tau_est = tau_est_vec.reshape(N, NUM_JOINTS)

    print(f"\n=== Métricas por joint — {tag} ===")
    for j in [4,5,6]:
        mse = mean_squared_error(tau[:, j], tau_est[:, j])
        r2  = r2_score(tau[:, j], tau_est[:, j])
        print(f"J{j+1}: MSE={mse:.6f}  R2={r2:.4f}  RMS(τ)={rms(tau[:,j]):.4f}")

        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(q[:, j]);  axs[0].set_ylabel(f"q{j+1} [rad]"); axs[0].grid(True)
        axs[1].plot(dq[:, j]); axs[1].set_ylabel(f"dq{j+1} [rad/s]"); axs[1].grid(True)
        axs[2].plot(tau[:, j], label="τ real")
        axs[2].plot(tau_est[:, j], "--", label="τ̂")
        axs[2].set_ylabel(f"τ{j+1} [Nm]"); axs[2].set_xlabel("muestra"); axs[2].grid(True); axs[2].legend()
        fig.suptitle(f"{tag} — Joint {j+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f"{tag}_J{j+1}.png"), dpi=120)
        plt.close(fig)

print("\n=== Energía (RMS) — señales articulares (TRAIN) ===")
for j in range(NUM_JOINTS):
    print(f"J{j+1}: RMS(dq)={rms(dq[:,j]):.5f}  RMS(ddq)={rms(ddq[:,j]):.5f}  RMS(τ)={rms(tau[:,j]):.5f}")

eval_and_plot(theta_full,  "FULL_WLS")
eval_and_plot(theta_J567,  "BLOCK_J5J6J7")
eval_and_plot(theta_J67,   "BLOCK_J6J7")
eval_and_plot(theta_J7,    "BLOCK_J7")

print(f"\n[OK] Resultados y gráficas en: {SAVE_DIR}")

