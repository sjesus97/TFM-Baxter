
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 0) CONFIG — PROCESO B (VALIDACIÓN MUÑECA)
# ============================================================
CSV_VAL = "wrist56_val_like_train_articular_butter_fc15_tauRAW.csv"

TRAIN_DIR = "/home/jesus/TFM_Baxter/work/ENTRENAMIENTO_DEFINITIVO/WRIST56_EXT_OFF_FC15"

THETA_FULL_FILE = os.path.join(TRAIN_DIR, "theta_WRIST_full.npy")
SIGMA_FILE      = os.path.join(TRAIN_DIR, "w_sigma_WRIST_full.npy")

THETA_J567_FILE = os.path.join(TRAIN_DIR, "theta_WRIST_BLOCK_J5J6J7.npy")
THETA_J67_FILE  = os.path.join(TRAIN_DIR, "theta_WRIST_BLOCK_J6J7.npy")
THETA_J7_FILE   = os.path.join(TRAIN_DIR, "theta_WRIST_BLOCK_J7.npy")

OUT_DIR_VAL = "/home/jesus/TFM_Baxter/work/VALIDACION_DEFINITIVA_WRIST56_EXT"
os.makedirs(OUT_DIR_VAL, exist_ok=True)

NUM_JOINTS = 7
EPS = 1e-12

# --- Modelo extendido B ---
NUM_INERTIAL_PER_JOINT = 10
NUM_FRIC_PER_JOINT     = 2
NUM_PARAMS_PER_JOINT   = NUM_INERTIAL_PER_JOINT + NUM_FRIC_PER_JOINT  # 12
P_BASE = NUM_JOINTS * NUM_PARAMS_PER_JOINT  # 84

ADD_OFFSETS = True
P_OFF = NUM_JOINTS if ADD_OFFSETS else 0

SPRING_JOINTS = []   
P_SPR = len(SPRING_JOINTS)

P_TOTAL = P_BASE + P_OFF + P_SPR  # 84 + 7 = 91

EPS_VEL = 0.03 

# checks
for f in [CSV_VAL, THETA_FULL_FILE, SIGMA_FILE, THETA_J567_FILE, THETA_J67_FILE, THETA_J7_FILE]:
    assert os.path.exists(f), f"No encuentro {f}"

# ============================================================
# 1) AUX (idénticas al TRAIN B)
# ============================================================
@jit(nopython=True)
def dotMat(v):
    return np.array([
        [v[0], -v[1], v[2], 0.0, 0.0, 0.0],
        [0.0,  v[0], 0.0, v[1], v[2], 0.0],
        [0.0, 0.0,  v[0], 0.0, v[1], v[2]],
    ])

@jit(nopython=True)
def crossMat(v):
    return np.array([
        [0.0, -v[2],  v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])

@jit(nopython=True)
def s(arr, j): return np.sin(arr[j])

@jit(nopython=True)
def c(arr, j): return np.cos(arr[j])

@jit(nopython=True)
def transMat(link, alphas, thetas):
    return np.array([
        [c(thetas, link), -s(thetas, link), 0.0],
        [s(thetas, link) * c(alphas, link),  c(thetas, link) * c(alphas, link), -s(alphas, link)],
        [s(thetas, link) * s(alphas, link),  c(thetas, link) * s(alphas, link),  c(alphas, link)],
    ])

# ============================================================
# 2) NEWTON–EULER -> obs (7,84) (idéntico a A/B)
# ============================================================
@jit(nopython=True)
def newtonEulerEstimate_ext(angles, velocities, accels, eps_vel):
    half_pi = np.pi / 2.0
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

    w_j_j = np.zeros((6, 12, 7))
    TMatrices = np.zeros((6, 6, 7))

    grav = np.array([0.0, 0.0, -9.8])

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
                                 + np.cross(ang_prev, velocities[link] * z + accels[link] * z)

            linearAccs[:, link]  = rotMat.T @ (
                linearAccs[:, link-1]
                + np.cross(angularAccs[:, link-1], pArray[:, link])
                + np.cross(angularVels[:, link-1], np.cross(angularVels[:, link-1], pArray[:, link]))
            )

        W_up = np.concatenate((
            (linearAccs[:, link] - grav).reshape(3, 1),
            crossMat(angularAccs[:, link]) + crossMat(angularVels[:, link]) @ crossMat(angularVels[:, link]),
            np.zeros((3, 8))
        ), axis=1)

        coul = np.tanh(velocities[link] / eps_vel)
        visc = velocities[link]

        W_lo = np.concatenate((
            np.zeros((3, 1)),
            crossMat(grav - linearAccs[:, link]),
            dotMat(angularAccs[:, link]) + crossMat(angularVels[:, link]) @ dotMat(angularVels[:, link]),
            np.array([[0.0, 0.0, coul]]).T,
            np.array([[0.0, 0.0, visc]]).T
        ), axis=1)

        w_j_j[:, :, link] = np.concatenate((W_up, W_lo), axis=0)

        T_up = np.concatenate((rotMat, np.zeros((3, 3))), axis=1)
        T_lo = np.concatenate((crossMat(pArray[:, link]) @ rotMat, rotMat), axis=1)
        TMatrices[:, :, link] = np.concatenate((T_up, T_lo), axis=0)

    w11,w22,w33,w44,w55,w66,w77 = w_j_j[:,:,0], w_j_j[:,:,1], w_j_j[:,:,2], w_j_j[:,:,3], w_j_j[:,:,4], w_j_j[:,:,5], w_j_j[:,:,6]
    T2,T3,T4,T5,T6,T7 = TMatrices[:,:,1], TMatrices[:,:,2], TMatrices[:,:,3], TMatrices[:,:,4], TMatrices[:,:,5], TMatrices[:,:,6]

    U77 = w77
    U66 = w66
    U67 = T7 @ w77
    U55 = w55
    U56 = T6 @ w66
    U57 = T6 @ T7 @ w77
    U44 = w44
    U45 = T5 @ w55
    U46 = T5 @ T6 @ w66
    U47 = T5 @ T6 @ T7 @ w77
    U33 = w33
    U34 = T4 @ w44
    U35 = T4 @ T5 @ w55
    U36 = T4 @ T5 @ T6 @ w66
    U37 = T4 @ T5 @ T6 @ T7 @ w77
    U22 = w22
    U23 = T3 @ w33
    U24 = T3 @ T4 @ w44
    U25 = T3 @ T4 @ T5 @ w55
    U26 = T3 @ T4 @ T5 @ T6 @ w66
    U27 = T3 @ T4 @ T5 @ T6 @ T7 @ w77
    U11 = w11
    U12 = T2 @ w22
    U13 = T2 @ T3 @ w33
    U14 = T2 @ T3 @ T4 @ w44
    U15 = T2 @ T3 @ T4 @ T5 @ w55
    U16 = T2 @ T3 @ T4 @ T5 @ T6 @ w66
    U17 = T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ w77

    obs = np.zeros((7, 84))
    li = -1

    def put_row(j, blocks):
        for (start, U) in blocks:
            obs[j, start:start+12] = U[li, :]

    put_row(0, [(0,U11),(12,U12),(24,U13),(36,U14),(48,U15),(60,U16),(72,U17)])
    put_row(1, [(12,U22),(24,U23),(36,U24),(48,U25),(60,U26),(72,U27)])
    put_row(2, [(24,U33),(36,U34),(48,U35),(60,U36),(72,U37)])
    put_row(3, [(36,U44),(48,U45),(60,U46),(72,U47)])
    put_row(4, [(48,U55),(60,U56),(72,U57)])
    put_row(5, [(60,U66),(72,U67)])
    put_row(6, [(72,U77)])
    return obs

# ============================================================
# utils
# ============================================================
def rms(x): return float(np.sqrt(np.mean(x**2)))

def row_idx_for_joint(j, N, n_joints=7):
    return np.arange(j, N*n_joints, n_joints)

def two_zoom_windows(dq_j, N, zoom_len):
    mid = N // 2
    def find_window(a, b):
        seg = dq_j[a:b]
        m = np.max(np.abs(seg))
        thr = 0.2*m if m > 0 else 0.0
        active = np.where(np.abs(seg) > thr)[0] if m > 0 else np.zeros(0, dtype=np.int64)
        start = a + active[0] if active.size > 0 else a
        end = min(start + zoom_len, b)
        return start, end
    return find_window(0, mid), find_window(mid, N)

# ============================================================
# 3) CARGA VALIDACIÓN
# ============================================================
print(f"[B-VAL] Leyendo {CSV_VAL} ...")
df = pd.read_csv(CSV_VAL)

q   = df[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
dq  = df[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
ddq = df[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
tau = df[[f"tau{j+1}" for j in range(NUM_JOINTS)]].to_numpy()

N = q.shape[0]
print(f"[B-VAL] Muestras: {N} | P_TOTAL={P_TOTAL} (base={P_BASE}, off={P_OFF}, spr={P_SPR}) | EPS_VEL={EPS_VEL}")

# ============================================================
# 4) Construir matObs_val (N*7 x P_TOTAL) y tau_vec_val
# ============================================================
matObs = np.zeros((N*NUM_JOINTS, P_TOTAL))
tau_vec = np.zeros(N*NUM_JOINTS)

col_off0 = P_BASE
col_spr0 = col_off0 + P_OFF
spring_map = {j: k for k, j in enumerate(SPRING_JOINTS)}

for i in range(N):
    obs84 = newtonEulerEstimate_ext(q[i], dq[i], ddq[i], EPS_VEL)  # (7,84)
    base = i * NUM_JOINTS
    for j in range(NUM_JOINTS):
        r = base + j
        matObs[r, :P_BASE] = obs84[j, :]
        tau_vec[r] = tau[i, j]

        if ADD_OFFSETS:
            matObs[r, col_off0 + j] = 1.0

        if j in spring_map:
            matObs[r, col_spr0 + spring_map[j]] = q[i, j]

print("[B-VAL] matObs:", matObs.shape, "tau_vec:", tau_vec.shape)

# ============================================================
# 5) Cargar thetas y sigma del TRAIN B
# ============================================================
theta_full = np.load(THETA_FULL_FILE).ravel()
theta_J567 = np.load(THETA_J567_FILE).ravel()
theta_J67  = np.load(THETA_J67_FILE).ravel()
theta_J7   = np.load(THETA_J7_FILE).ravel()
sigma_train = np.load(SIGMA_FILE).ravel()

for name, th in [("FULL", theta_full), ("J567", theta_J567), ("J67", theta_J67), ("J7", theta_J7)]:
    assert th.shape[0] == P_TOTAL, f"theta_{name} tiene {th.shape[0]} params, pero P_TOTAL={P_TOTAL}"
assert sigma_train.shape[0] == NUM_JOINTS

print("[B-VAL] sigma_train:", sigma_train.round(6))

# ============================================================
# 6) Evaluación (SOLO J5–J7) + plots + zooms (q,dq,tau)
# ============================================================
def eval_theta(theta_vec, tag):
    tau_hat = (matObs @ theta_vec).reshape(N, NUM_JOINTS)

    print(f"\n[B-VAL] === Métricas VALIDACIÓN — {tag} (SOLO J5–J7) ===")
    ZOOM_LEN = min(2000, N)

    for j in [4, 5, 6]:
        tau_j = tau[:, j]
        tau_h = tau_hat[:, j]
        dq_j  = dq[:, j]

        rmse = float(np.sqrt(mean_squared_error(tau_j, tau_h)))
        rms_tau = rms(tau_j)
        rmse_rel = 100.0 * rmse / (rms_tau + EPS)
        r2 = float(r2_score(tau_j, tau_h))

        # centrado para corr
        tj = tau_j - np.mean(tau_j)
        th = tau_h - np.mean(tau_h)
        corr = float(np.corrcoef(tj, th)[0, 1]) if (np.std(tj)>1e-9 and np.std(th)>1e-9) else np.nan

        # R2 en zona activa (por |dq|)
        max_abs = float(np.max(np.abs(dq_j)))
        thr = 0.2 * max_abs if max_abs > 0 else 0.0
        mask_act = np.abs(dq_j) > thr
        r2_act = float(r2_score(tau_j[mask_act], tau_h[mask_act])) if np.any(mask_act) else np.nan

        print(f"  J{j+1}: RMSE={rmse:.4f} Nm | RMSE_rel={rmse_rel:.2f}% | R2={r2:.4f} | R2_activo={r2_act:.4f} | corr={corr:.4f} | RMS(τ)={rms_tau:.4f}")

        # -------- Global τ vs τ̂ --------
        plt.figure(figsize=(10, 4))
        plt.plot(tau_j, label="τ real")
        plt.plot(tau_h, "--", label=f"τ̂ {tag}")
        plt.title(f"B-VALIDACIÓN — {tag} Joint {j+1} (GLOBAL)")
        plt.xlabel("muestra"); plt.ylabel("Torque [Nm]")
        plt.grid(True); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_VAL, f"VAL_{tag}_GLOBAL_J{j+1}.png"), dpi=130)
        plt.close()

        # -------- ZOOMS: q, dq, τ vs τ̂ --------
        (s1, e1), (s2, e2) = two_zoom_windows(dq_j, N, ZOOM_LEN)
        for (start, end, phase) in [(s1, e1, "PHASE1"), (s2, e2, "PHASE2")]:
            x = np.arange(start, end)

            fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
            axs[0].plot(x, q[start:end, j]);  axs[0].set_ylabel(f"q{j+1} [rad]");   axs[0].grid(True)
            axs[1].plot(x, dq[start:end, j]); axs[1].set_ylabel(f"dq{j+1} [rad/s]");axs[1].grid(True)
            axs[2].plot(x, tau[start:end, j], label="τ real")
            axs[2].plot(x, tau_hat[start:end, j], "--", label=f"τ̂ {tag}")
            axs[2].set_ylabel(f"τ{j+1} [Nm]"); axs[2].set_xlabel("muestra")
            axs[2].grid(True); axs[2].legend()

            fig.suptitle(f"B-VALIDACIÓN — {tag} Joint {j+1} ({phase} ZOOM)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR_VAL, f"VAL_{tag}_ZOOM_{phase}_J{j+1}.png"), dpi=130)
            plt.close(fig)

    return tau_hat

tau_hat_full = eval_theta(theta_full, "B_FULL")
eval_theta(theta_J567, "B_BLOCK_J5J6J7")
eval_theta(theta_J67,  "B_BLOCK_J6J7")
eval_theta(theta_J7,   "B_BLOCK_J7")

# ============================================================
# 7) Diagnóstico WLS con sigma del TRAIN B (opcional pero útil)
# ============================================================
w = np.zeros(N*NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = np.arange(j, N*NUM_JOINTS, NUM_JOINTS)
    w[idx] = 1.0 / (sigma_train[j] + EPS)

Wsqrt = np.sqrt(w)
A_val_w = matObs * Wsqrt[:, None]
b_val_w = tau_vec * Wsqrt

mse_w_global = float(np.mean((A_val_w @ theta_full - b_val_w)**2))
print(f"\n[B-VAL] MSE ponderado global (WLS diag, usando theta B_FULL): {mse_w_global:.6f}")

print("[B-VAL] MSE ponderado por joint (WLS diag, theta B_FULL):")
for j in [4, 5, 6]:
    err = (tau_hat_full[:, j] - tau[:, j]) / (sigma_train[j] + EPS)
    print(f"  J{j+1}: {float(np.mean(err**2)):.6f}")

# ============================================================
# 8) Resumen final
# ============================================================
print(f"\n[B-VAL] [OK] Resultados guardados en: {OUT_DIR_VAL}")

