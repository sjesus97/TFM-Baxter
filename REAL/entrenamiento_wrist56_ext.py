
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================
# 0) CONFIG — PROCESO B (MUÑECA)
# ============================================================
CSV_INPUT = "wrist56_train_basic_articular_butter_fc15_tauRAW.csv"

OUT_DIR_TRAIN = "/home/jesus/TFM_Baxter/work/ENTRENAMIENTO_DEFINITIVO/WRIST56_EXT_OFF_FC15"
os.makedirs(OUT_DIR_TRAIN, exist_ok=True)

NUM_JOINTS = 7
EPS = 1e-12

# --- Modelo extendido-
NUM_INERTIAL_PER_JOINT = 10
NUM_FRIC_PER_JOINT     = 2
NUM_PARAMS_PER_JOINT   = NUM_INERTIAL_PER_JOINT + NUM_FRIC_PER_JOINT  # 12
P_BASE = NUM_JOINTS * NUM_PARAMS_PER_JOINT  # 84

ADD_OFFSETS = True
P_OFF = NUM_JOINTS if ADD_OFFSETS else 0


SPRING_JOINTS = []          
P_SPR = len(SPRING_JOINTS)

P_TOTAL = P_BASE + P_OFF + P_SPR  # 84 + 7 = 91

# Archivos
SAVE_THETA_FULL = os.path.join(OUT_DIR_TRAIN, "theta_WRIST_full.npy")
SAVE_WSIGMA     = os.path.join(OUT_DIR_TRAIN, "w_sigma_WRIST_full.npy")

SAVE_THETA_J567 = os.path.join(OUT_DIR_TRAIN, "theta_WRIST_BLOCK_J5J6J7.npy")
SAVE_THETA_J67  = os.path.join(OUT_DIR_TRAIN, "theta_WRIST_BLOCK_J6J7.npy")
SAVE_THETA_J7   = os.path.join(OUT_DIR_TRAIN, "theta_WRIST_BLOCK_J7.npy")

# Fricción Coulomb suave
EPS_VEL = 0.03  # rad/s (mantén el mismo que en A)

RIDGE_LAMBDA = 1e-6

# ============================================================
# 1) AUX
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
# 2) NEWTON–EULER -> obs (7, 84) (idéntico a A)
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

def plot_qdq_tau_triplet(qv, dqv, tau_true, tau_hat, title, savepath, x=None):
    """3 subplots: q, dq, tau vs tau_hat. x opcional para zoom."""
    if x is None:
        x = np.arange(len(qv))
    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(x, qv);   axs[0].set_ylabel("q [rad]");     axs[0].grid(True)
    axs[1].plot(x, dqv);  axs[1].set_ylabel("dq [rad/s]");  axs[1].grid(True)
    axs[2].plot(x, tau_true, label="τ real")
    axs[2].plot(x, tau_hat, "--", label="τ̂")
    axs[2].set_ylabel("τ [Nm]"); axs[2].set_xlabel("muestra")
    axs[2].grid(True); axs[2].legend()
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(savepath, dpi=130)
    plt.close(fig)

# ============================================================
# 3) CARGA
# ============================================================
print(f"[B-TRAIN] Leyendo {CSV_INPUT} ...")
df = pd.read_csv(CSV_INPUT)

q   = df[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
dq  = df[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
ddq = df[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
tau = df[[f"tau{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
N = q.shape[0]
print(f"[B-TRAIN] Muestras: {N}")
print(f"[B-TRAIN] P_TOTAL = {P_TOTAL} (base={P_BASE}, off={P_OFF}, spr={P_SPR}) | EPS_VEL={EPS_VEL} | RIDGE={RIDGE_LAMBDA}")

# ============================================================
# 4) Construir matObs (N*7 x P_TOTAL) y tau_vec
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

print("[B-TRAIN] matObs:", matObs.shape, "tau_vec:", tau_vec.shape)

# ============================================================
# 5) WLS por articulación (igual que A)
# ============================================================
sigma = np.zeros(NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = np.arange(j, N*NUM_JOINTS, NUM_JOINTS)
    sigma[j] = rms(tau_vec[idx]) + EPS

w = np.zeros(N*NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = np.arange(j, N*NUM_JOINTS, NUM_JOINTS)
    w[idx] = 1.0 / sigma[j]

Wsqrt = np.sqrt(w)
A = matObs * Wsqrt[:, None]
b = tau_vec * Wsqrt

np.save(SAVE_WSIGMA, sigma)
print("[B-TRAIN] Escalas por joint (RMS):", sigma.round(6))

# ============================================================
# 6) REGRESIÓN FULL (B-FULL)
# ============================================================
if RIDGE_LAMBDA > 0.0:
    AtA = A.T @ A
    Atb = A.T @ b
    theta_full = np.linalg.solve(AtA + RIDGE_LAMBDA*np.eye(P_TOTAL), Atb)
    rank = np.linalg.matrix_rank(AtA)
else:
    theta_full, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

np.save(SAVE_THETA_FULL, theta_full)
print(f"[B-TRAIN] theta_full guardado en {SAVE_THETA_FULL}")
print(f"[B-TRAIN] Rango(A) = {rank} / {P_TOTAL}")

# ============================================================
# 7) Métricas TRAIN (B-FULL) — reportamos J5–J7 + plots completos
# ============================================================
tau_est_full = (matObs @ theta_full).reshape(N, NUM_JOINTS)

print("\n[B-TRAIN] === Métricas B-FULL (reportamos SOLO J5–J7) ===")
ZOOM_LEN = min(2000, N)

for j in [4, 5, 6]:
    tau_j = tau[:, j]
    tau_hat = tau_est_full[:, j]
    dq_j = dq[:, j]

    rmse = float(np.sqrt(mean_squared_error(tau_j, tau_hat)))
    r2 = float(r2_score(tau_j, tau_hat))

    tau_j_c = tau_j - np.mean(tau_j)
    tau_hat_c = tau_hat - np.mean(tau_hat)
    corr = float(np.corrcoef(tau_j_c, tau_hat_c)[0, 1]) if (np.std(tau_j_c)>1e-9 and np.std(tau_hat_c)>1e-9) else np.nan

    print(f"  J{j+1}: RMSE={rmse:.4f} Nm | R2={r2:.4f} | corr={corr:.4f} | RMS(τ)={rms(tau_j):.4f}")

    # (1) plot global τ vs τ̂ (se queda)
    plt.figure(figsize=(10, 4))
    plt.plot(tau_j, label="τ real")
    plt.plot(tau_hat, "--", label="τ̂ B-FULL")
    plt.title(f"B-FULL — Joint {j+1} (TRAIN) — τ vs τ̂")
    plt.xlabel("muestra"); plt.ylabel("Torque [Nm]")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_TRAIN, f"TRAIN_BFULL_J{j+1}_TAUONLY.png"), dpi=130)
    plt.close()

    # (2) plot global completo q, dq, τ
    plot_qdq_tau_triplet(
        q[:, j], dq[:, j], tau[:, j], tau_est_full[:, j],
        title=f"B-FULL — Joint {j+1} (TRAIN) — q, dq, τ vs τ̂",
        savepath=os.path.join(OUT_DIR_TRAIN, f"TRAIN_BFULL_J{j+1}_QDQTAU_GLOBAL.png"),
        x=np.arange(N)
    )

    # zooms (q,dq,τ)
    (s1, e1), (s2, e2) = two_zoom_windows(dq_j, N, ZOOM_LEN)
    for (start, end, tag_phase) in [(s1, e1, "PHASE1"), (s2, e2, "PHASE2")]:
        x = np.arange(start, end)

        # (3) zoom completo q, dq, τ
        plot_qdq_tau_triplet(
            q[start:end, j], dq[start:end, j],
            tau[start:end, j], tau_est_full[start:end, j],
            title=f"B-FULL — Joint {j+1} ({tag_phase} ZOOM) — q, dq, τ vs τ̂",
            savepath=os.path.join(OUT_DIR_TRAIN, f"TRAIN_BFULL_J{j+1}_QDQTAU_ZOOM_{tag_phase}.png"),
            x=x
        )

# ============================================================
# 8) BLOQUES MUÑECA (B-BLOCK): J7, J6-7, J5-7
# ============================================================
print("\n[B-TRAIN] ================= DESACOPLE MUÑECA (B-BLOCK) =================")

COLS_BASE_J5 = slice(48, 84)
COLS_BASE_J6 = slice(60, 84)
COLS_BASE_J7 = slice(72, 84)

col_off0 = P_BASE

COLS_J7   = np.array(list(range(COLS_BASE_J7.start,  COLS_BASE_J7.stop)) + [col_off0 + 6], dtype=np.int64)
COLS_J67  = np.array(list(range(COLS_BASE_J6.start,  COLS_BASE_J6.stop)) + [col_off0 + 5, col_off0 + 6], dtype=np.int64)
COLS_J567 = np.array(list(range(COLS_BASE_J5.start,  COLS_BASE_J5.stop)) + [col_off0 + 4, col_off0 + 5, col_off0 + 6], dtype=np.int64)

ROWS_J7   = row_idx_for_joint(6, N, NUM_JOINTS)
ROWS_J6   = row_idx_for_joint(5, N, NUM_JOINTS)
ROWS_J5   = row_idx_for_joint(4, N, NUM_JOINTS)
ROWS_J67  = np.sort(np.concatenate([ROWS_J6, ROWS_J7]))
ROWS_J567 = np.sort(np.concatenate([ROWS_J5, ROWS_J6, ROWS_J7]))

def fit_block_padded(rows, cols, ridge_lambda=0.0):
    A_block = matObs[rows[:, None], cols[None, :]]
    b_block = tau_vec[rows]

    sigma_row = np.empty(rows.shape[0], dtype=float)
    for k in range(rows.shape[0]):
        sigma_row[k] = sigma[rows[k] % NUM_JOINTS]
    Wsqrt_loc = 1.0 / np.sqrt(sigma_row + 1e-12)

    A_w = A_block * Wsqrt_loc[:, None]
    b_w = b_block * Wsqrt_loc

    col_rms = np.sqrt(np.mean(A_w**2, axis=0)) + 1e-12
    A_wn = A_w / col_rms[None, :]

    if ridge_lambda > 0.0:
        AtA = A_wn.T @ A_wn
        Atb = A_wn.T @ b_w
        theta_n = np.linalg.solve(AtA + ridge_lambda*np.eye(A_wn.shape[1]), Atb)
        rank_loc = np.linalg.matrix_rank(AtA)
    else:
        theta_n, _, rank_loc, _ = np.linalg.lstsq(A_wn, b_w, rcond=None)

    theta_loc = theta_n / col_rms

    theta_pad = np.zeros(P_TOTAL)
    theta_pad[cols] = theta_loc

    try:
        S = np.linalg.svd(A_block, full_matrices=False)[1]
        cond = (S.max() / S.min()) if np.all(S > 1e-12) else np.inf
    except Exception:
        cond = np.nan

    return theta_pad, rank_loc, cond

theta_J7,   rank_J7,   cond_J7   = fit_block_padded(ROWS_J7,   COLS_J7,   ridge_lambda=0.0)
theta_J67,  rank_J67,  cond_J67  = fit_block_padded(ROWS_J67,  COLS_J67,  ridge_lambda=0.0)
theta_J567, rank_J567, cond_J567 = fit_block_padded(ROWS_J567, COLS_J567, ridge_lambda=0.0)

np.save(SAVE_THETA_J7, theta_J7)
np.save(SAVE_THETA_J67, theta_J67)
np.save(SAVE_THETA_J567, theta_J567)

print(f"[B-TRAIN] [OK] θ_BLOCK_J7   → {SAVE_THETA_J7}     rank={rank_J7}   cond={cond_J7:.3e}")
print(f"[B-TRAIN] [OK] θ_BLOCK_J67  → {SAVE_THETA_J67}   rank={rank_J67}  cond={cond_J67:.3e}")
print(f"[B-TRAIN] [OK] θ_BLOCK_J567 → {SAVE_THETA_J567} rank={rank_J567} cond={cond_J567:.3e}")

# ============================================================
# 9) Eval comparativa TRAIN (J5–J7) para FULL vs BLOCKS
#     + plots completos (q,dq,tau) + zooms
# ============================================================
def eval_model(theta_vec, tag):
    tau_hat = (matObs @ theta_vec).reshape(N, NUM_JOINTS)
    print(f"\n[B-TRAIN] === {tag} (TRAIN, SOLO J5–J7) ===")

    for j in [4, 5, 6]:
        rmse = float(np.sqrt(mean_squared_error(tau[:, j], tau_hat[:, j])))
        r2 = float(r2_score(tau[:, j], tau_hat[:, j]))

        tj = tau[:, j] - np.mean(tau[:, j])
        th = tau_hat[:, j] - np.mean(tau_hat[:, j])
        corr = float(np.corrcoef(tj, th)[0, 1]) if (np.std(tj)>1e-9 and np.std(th)>1e-9) else np.nan

        print(f"  J{j+1}: RMSE={rmse:.4f} | R2={r2:.4f} | corr={corr:.4f}")

        # (1) global completo q,dq,tau
        plot_qdq_tau_triplet(
            q[:, j], dq[:, j], tau[:, j], tau_hat[:, j],
            title=f"{tag} — Joint {j+1} (TRAIN) — q, dq, τ vs τ̂",
            savepath=os.path.join(OUT_DIR_TRAIN, f"TRAIN_{tag}_J{j+1}_QDQTAU_GLOBAL.png"),
            x=np.arange(N)
        )

        # (2) zooms completos
        dq_j = dq[:, j]
        (s1, e1), (s2, e2) = two_zoom_windows(dq_j, N, ZOOM_LEN)
        for (start, end, tag_phase) in [(s1, e1, "PHASE1"), (s2, e2, "PHASE2")]:
            x = np.arange(start, end)
            plot_qdq_tau_triplet(
                q[start:end, j], dq[start:end, j],
                tau[start:end, j], tau_hat[start:end, j],
                title=f"{tag} — Joint {j+1} ({tag_phase} ZOOM) — q, dq, τ vs τ̂",
                savepath=os.path.join(OUT_DIR_TRAIN, f"TRAIN_{tag}_J{j+1}_QDQTAU_ZOOM_{tag_phase}.png"),
                x=x
            )

eval_model(theta_full, "B_FULL")
eval_model(theta_J567, "B_BLOCK_J5J6J7")
eval_model(theta_J67,  "B_BLOCK_J6J7")
eval_model(theta_J7,   "B_BLOCK_J7")

print(f"\n[B-TRAIN] [OK] Todo guardado en: {OUT_DIR_TRAIN}")

