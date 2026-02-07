
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 0) CONFIG
# ===============================
CSV_INPUT = "data_train_phi75_theta145_art_def_butter_fc15.csv"

OUT_DIR_TRAIN = "/home/jesus/TFM_Baxter/work/ENTRENAMIENTO_DEFINITIVO/FRICCION_SPR_J1J2_DEF/MUÑECA_DESACOP"
os.makedirs(OUT_DIR_TRAIN, exist_ok=True)

NUM_JOINTS = 7
EPS = 1e-12

# --- Modelo extendido ---
NUM_INERTIAL_PER_JOINT = 10
NUM_FRIC_PER_JOINT     = 2
NUM_PARAMS_PER_JOINT   = NUM_INERTIAL_PER_JOINT + NUM_FRIC_PER_JOINT  # 12
P_BASE = NUM_JOINTS * NUM_PARAMS_PER_JOINT  # 84

# offsets por joint (7)
ADD_OFFSETS = True
P_OFF = NUM_JOINTS if ADD_OFFSETS else 0

# muelles en J1 y J2
SPRING_JOINTS = [0, 1]       # J1 + J2
P_SPR = len(SPRING_JOINTS)

P_TOTAL = P_BASE + P_OFF + P_SPR  # 84 + 7 + 2 = 93

SAVE_THETA_EXT = os.path.join(OUT_DIR_TRAIN, "theta_EXT_full.npy")
SAVE_WSIGMA    = os.path.join(OUT_DIR_TRAIN, "w_sigma_EXT_full.npy")

# >>> NUEVO: thetas específicas de muñeca (acolchadas a P_TOTAL)
SAVE_THETA_J567 = os.path.join(OUT_DIR_TRAIN, "theta_EXT_BLOCK_J5J6J7.npy")
SAVE_THETA_J67  = os.path.join(OUT_DIR_TRAIN, "theta_EXT_BLOCK_J6J7.npy")
SAVE_THETA_J7   = os.path.join(OUT_DIR_TRAIN, "theta_EXT_BLOCK_J7.npy")

# Coulomb suave tanh(dq/eps)
EPS_VEL = 0.03  # rad/s

# Ridge opcional
RIDGE_LAMBDA = 1e-6

# ===============================
# 1) AUX
# ===============================
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

# ===============================
# 2) NEWTON–EULER -> obs (7, 84)
# ===============================
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

# ---------- utilidades ----------
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
# 4) CONSTRUIR matObs (N*7 x P_TOTAL) y tau_vec
# ===============================
print(f"Parámetros: base={P_BASE}, offsets={P_OFF}, muelles={P_SPR} -> TOTAL={P_TOTAL}")

matObs = np.zeros((N*NUM_JOINTS, P_TOTAL))
tau_vec  = np.zeros(N*NUM_JOINTS)

col_off0 = P_BASE
col_spr0 = col_off0 + P_OFF
spring_map = {j: k for k, j in enumerate(SPRING_JOINTS)}  # joint -> local spring idx

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

print("matObs:", matObs.shape, "tau_vec:", tau_vec.shape)

# ===============================
# 5) WLS por articulación
# ===============================
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
print("Escalas por joint (RMS):", sigma.round(6))

# ===============================
# 6) REGRESIÓN GLOBAL (FULL)
# ===============================
if RIDGE_LAMBDA > 0.0:
    AtA = A.T @ A
    Atb = A.T @ b
    theta_ext = np.linalg.solve(AtA + RIDGE_LAMBDA*np.eye(P_TOTAL), Atb)
    rank = np.linalg.matrix_rank(AtA)
else:
    theta_ext, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)

np.save(SAVE_THETA_EXT, theta_ext)
print(f"theta_ext (FULL) guardado en {SAVE_THETA_EXT}")
print(f"Rango(A) = {rank} / {P_TOTAL}")
print(f"EPS_VEL={EPS_VEL} rad/s  |  RIDGE_LAMBDA={RIDGE_LAMBDA}")

# ===============================
# 7) PREDICCIÓN GLOBAL + plots (FULL)
# ===============================
tau_est_vec = matObs @ theta_ext
tau_est = tau_est_vec.reshape(N, NUM_JOINTS)

print("\n=== Métricas por joint (FULL: rígido + fricción + muelle + offset) ===")
ZOOM_LEN = min(2000, N)

for j in range(NUM_JOINTS):
    tau_j = tau[:, j]
    tau_hat_j = tau_est[:, j]
    dq_j = dq[:, j]

    mse = mean_squared_error(tau_j, tau_hat_j)
    rmse = np.sqrt(mse)
    rms_tau_j = rms(tau_j)
    rmse_rel = 100.0 * rmse / (rms_tau_j + EPS)
    r2 = r2_score(tau_j, tau_hat_j)

    tau_j_c = tau_j - np.mean(tau_j)
    tau_hat_j_c = tau_hat_j - np.mean(tau_hat_j)
    mse_c = mean_squared_error(tau_j_c, tau_hat_j_c)
    rmse_c = np.sqrt(mse_c)
    var_tau_c = np.var(tau_j_c) + EPS
    r2_centered = 1.0 - mse_c / var_tau_c
    corr = float(np.corrcoef(tau_j_c, tau_hat_j_c)[0, 1]) if (np.std(tau_j_c)>1e-9 and np.std(tau_hat_j_c)>1e-9) else np.nan

    max_abs_dq = np.max(np.abs(dq_j))
    thr = 0.2 * max_abs_dq if max_abs_dq > 0 else 0.0
    mask_act = np.abs(dq_j) > thr
    r2_act = r2_score(tau_j[mask_act], tau_hat_j[mask_act]) if np.any(mask_act) else np.nan

    print(
        f"Joint {j+1}: RMSE={rmse:.4f} Nm, RMSE_rel={rmse_rel:.2f}%, "
        f"R2={r2:.4f}, R2_activo={r2_act:.4f}, R2_centrado={r2_centered:.4f}, corr={corr:.4f}"
    )

    plt.figure(figsize=(10, 4))
    plt.plot(tau_j, label=f"τ real J{j+1}")
    plt.plot(tau_hat_j, "--", label=f"τ estimado J{j+1}")
    plt.title(f"FULL — Joint {j+1}")
    plt.xlabel("muestra")
    plt.ylabel("Torque [Nm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_TRAIN, f"FULL_GLOBAL_J{j+1}.png"), dpi=130)
    plt.close()

    (start1, end1), (start2, end2) = two_zoom_windows(dq_j, N, ZOOM_LEN)
    for (start, end, tag_phase) in [(start1,end1,"PHASE1_RANDOM"), (start2,end2,"PHASE2_SPIRAL")]:
        x_z = np.arange(start, end)
        q_z = q[start:end, j]
        dq_z = dq[start:end, j]
        tau_z = tau[start:end, j]
        tau_est_z = tau_est[start:end, j]

        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(x_z, q_z);  axs[0].set_ylabel(f"q{j+1} [rad]");   axs[0].grid(True)
        axs[1].plot(x_z, dq_z); axs[1].set_ylabel(f"dq{j+1} [rad/s]");axs[1].grid(True)
        axs[2].plot(x_z, tau_z, label="τ real")
        axs[2].plot(x_z, tau_est_z, "--", label="τ̂ FULL")
        axs[2].set_ylabel(f"τ{j+1} [Nm]"); axs[2].set_xlabel("muestra"); axs[2].grid(True); axs[2].legend()
        fig.suptitle(f"FULL — Joint {j+1} ({tag_phase} ZOOM)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_TRAIN, f"FULL_GLOBAL_J{j+1}_ZOOM_{tag_phase}.png"), dpi=120)
        plt.close(fig)

# ============================================================
# 7.5 MODELO REDUCIDO PARA J1
# ============================================================
print("\n=========== MODELO REDUCIDO PARA J1 (c(t)*θ + k_v*dq + b*tanh + k_s*q + d) ===========")

rows_J1 = row_idx_for_joint(0, N, NUM_JOINTS)
cols_J1_inert = np.arange(0, 12)  # bloque de J1 dentro del base

YJ1 = matObs[rows_J1, :P_BASE][:, cols_J1_inert]
col_rms_inert = np.sqrt(np.mean(YJ1[:, 0:10]**2, axis=0))
idx_live = int(np.argmax(col_rms_inert))
c_t = YJ1[:, idx_live]

dq1 = dq[:, 0]
tau1 = tau[:, 0]
coul = np.tanh(dq1 / EPS_VEL)
q1 = q[:, 0]

use_spring_J1 = (0 in SPRING_JOINTS)
if use_spring_J1:
    X_red = np.column_stack([c_t, dq1, coul, q1, np.ones_like(c_t)])
else:
    X_red = np.column_stack([c_t, dq1, coul, np.ones_like(c_t)])

theta_red, *_ = np.linalg.lstsq(X_red, tau1, rcond=None)
tau1_red = X_red @ theta_red

mse_red  = mean_squared_error(tau1, tau1_red)
rmse_red = np.sqrt(mse_red)
r2_red   = r2_score(tau1, tau1_red)

print("Parámetros reducidos J1:")
if use_spring_J1:
    print(f"  θ_eff(inercial) = {theta_red[0]:+.5f}")
    print(f"  k_visc          = {theta_red[1]:+.5f}")
    print(f"  b_coul          = {theta_red[2]:+.5f}")
    print(f"  k_spring        = {theta_red[3]:+.5f}")
    print(f"  d_offset        = {theta_red[4]:+.5f}")
else:
    print(f"  θ_eff(inercial) = {theta_red[0]:+.5f}")
    print(f"  k_visc          = {theta_red[1]:+.5f}")
    print(f"  b_coul          = {theta_red[2]:+.5f}")
    print(f"  d_offset        = {theta_red[3]:+.5f}")

print(f"Métricas reducido J1: RMSE={rmse_red:.4f} Nm, R2={r2_red:.4f}")

plt.figure(figsize=(10,5))
plt.plot(tau1, label="τ1 real")
plt.plot(tau1_red, '--', label="τ1 estimado (reducido)")
plt.title("REDUCED J1 — τ vs τ̂")
plt.xlabel("muestra"); plt.ylabel("Torque [Nm]")
plt.grid(True); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_TRAIN, "REDUCED_J1_EXT.png"), dpi=140)
plt.close()

(start1, end1), (start2, end2) = two_zoom_windows(dq1, N, ZOOM_LEN)
for (start, end, tag_phase) in [(start1,end1,"PHASE1_RANDOM"), (start2,end2,"PHASE2_SPIRAL")]:
    x_z = np.arange(start, end)
    q1_z = q[start:end, 0]
    dq1_z = dq[start:end, 0]
    tau1_z = tau[start:end, 0]
    tau1_red_z = tau1_red[start:end]

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axs[0].plot(x_z, q1_z);  axs[0].set_ylabel("q1 [rad]");      axs[0].grid(True)
    axs[1].plot(x_z, dq1_z); axs[1].set_ylabel("dq1 [rad/s]");   axs[1].grid(True)
    axs[2].plot(x_z, tau1_z, label="τ1 real")
    axs[2].plot(x_z, tau1_red_z, "--", label="τ̂1 reducido")
    axs[2].set_ylabel("τ1 [Nm]"); axs[2].set_xlabel("muestra")
    axs[2].grid(True); axs[2].legend()
    fig.suptitle(f"REDUCED J1 — {tag_phase} ZOOM")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_TRAIN, f"REDUCED_J1_EXT_ZOOM_{tag_phase}.png"), dpi=130)
    plt.close(fig)

# ===============================
# 8) BLOQUES MUÑECA (DESACOPLE) — versión EXT (P_TOTAL=93)
# ===============================
print("\n================= DESACOPLE MUÑECA (EXT) =================")

# Columnas base por joint:
# J1: 0..11, J2:12..23, ..., J7:72..83
COLS_BASE_J5   = slice(48, 84)   # J5..J7 dentro del base (48..83)
COLS_BASE_J6   = slice(60, 84)   # J6..J7
COLS_BASE_J7   = slice(72, 84)   # J7

# Offsets (7) empiezan en col_off0 = 84:
col_off0 = P_BASE

# Springs (2) empiezan en col_spr0 = 91:
col_spr0 = P_BASE + P_OFF

# Para muñeca queremos: base(J5..), offsets de J5-7, fricción ya va dentro del base
# Springs NO: porque son J1/J2.
COLS_OFF_J5J6J7 = [col_off0 + 4, col_off0 + 5, col_off0 + 6]

# Construimos listas de columnas para cada bloque:
COLS_J7   = list(range(COLS_BASE_J7.start,  COLS_BASE_J7.stop))  + [col_off0 + 6]
COLS_J67  = list(range(COLS_BASE_J6.start,  COLS_BASE_J6.stop))  + [col_off0 + 5, col_off0 + 6]
COLS_J567 = list(range(COLS_BASE_J5.start,  COLS_BASE_J5.stop))  + COLS_OFF_J5J6J7

COLS_J7   = np.array(COLS_J7, dtype=np.int64)
COLS_J67  = np.array(COLS_J67, dtype=np.int64)
COLS_J567 = np.array(COLS_J567, dtype=np.int64)

ROWS_J7   = row_idx_for_joint(6, N, NUM_JOINTS)
ROWS_J6   = row_idx_for_joint(5, N, NUM_JOINTS)
ROWS_J5   = row_idx_for_joint(4, N, NUM_JOINTS)
ROWS_J67  = np.sort(np.concatenate([ROWS_J6, ROWS_J7]))
ROWS_J567 = np.sort(np.concatenate([ROWS_J5, ROWS_J6, ROWS_J7]))

def fit_block_padded_ext(rows, cols, ridge_lambda=0.0):
    """
    Ajusta un bloque en (rows, cols) con WLS fila-a-fila y devuelve theta acolchado (P_TOTAL,).
    Mantiene la lógica de tu script antiguo, adaptada a P_TOTAL.
    """
    A_block = matObs[rows[:, None], cols[None, :]]  # (nrows, ncols)
    b_block = tau_vec[rows]

    # WLS local: sigma depende del joint (r % 7)
    sigma_row = np.empty(rows.shape[0], dtype=float)
    for k in range(rows.shape[0]):
        sigma_row[k] = sigma[rows[k] % NUM_JOINTS]
    Wsqrt_loc = 1.0 / np.sqrt(sigma_row + 1e-12)

    A_w = A_block * Wsqrt_loc[:, None]
    b_w = b_block * Wsqrt_loc

    # normalización de columnas
    col_rms = np.sqrt(np.mean(A_w**2, axis=0)) + 1e-12
    A_wn = A_w / col_rms[None, :]

    # solve (LS o Ridge)
    if ridge_lambda > 0.0:
        AtA = A_wn.T @ A_wn
        Atb = A_wn.T @ b_w
        theta_local_n = np.linalg.solve(AtA + ridge_lambda*np.eye(A_wn.shape[1]), Atb)
        rank_loc = np.linalg.matrix_rank(AtA)
    else:
        theta_local_n, _, rank_loc, _ = np.linalg.lstsq(A_wn, b_w, rcond=None)

    theta_local = theta_local_n / col_rms

    theta_pad = np.zeros(P_TOTAL)
    theta_pad[cols] = theta_local

    # cond diagnóstico (sin pesos, opcional)
    try:
        S = np.linalg.svd(A_block, full_matrices=False)[1]
        cond = (S.max() / S.min()) if np.all(S > 1e-12) else np.inf
    except Exception:
        cond = np.nan

    return theta_pad, rank_loc, cond

theta_J7,   rank_J7,   cond_J7   = fit_block_padded_ext(ROWS_J7,   COLS_J7,   ridge_lambda=0.0)
theta_J67,  rank_J67,  cond_J67  = fit_block_padded_ext(ROWS_J67,  COLS_J67,  ridge_lambda=0.0)
theta_J567, rank_J567, cond_J567 = fit_block_padded_ext(ROWS_J567, COLS_J567, ridge_lambda=0.0)

np.save(SAVE_THETA_J7,   theta_J7)
np.save(SAVE_THETA_J67,  theta_J67)
np.save(SAVE_THETA_J567, theta_J567)

print(f"[OK] θ_BLOCK_J7   → {SAVE_THETA_J7}     rank={rank_J7}   cond={cond_J7:.3e}")
print(f"[OK] θ_BLOCK_J67  → {SAVE_THETA_J67}   rank={rank_J67}  cond={cond_J67:.3e}")
print(f"[OK] θ_BLOCK_J567 → {SAVE_THETA_J567} rank={rank_J567} cond={cond_J567:.3e}")

# ===============================
# 9) Métricas TRAIN para modelos de muñeca (sólo J5–J7)
# ===============================
def eval_block_train_ext(theta_vec, tag):
    tau_est_vec_block = matObs @ theta_vec
    tau_est_block = tau_est_vec_block.reshape(N, NUM_JOINTS)

    print(f"\n=== Métricas por joint — {tag} (TRAIN, mostramos J5–J7) ===")
    for j in [4, 5, 6]:
        tau_j = tau[:, j]
        tau_hat_j = tau_est_block[:, j]
        dq_j_all = dq[:, j]

        mse = mean_squared_error(tau_j, tau_hat_j)
        rmse = np.sqrt(mse)
        r2  = r2_score(tau_j, tau_hat_j)

        tau_j_c = tau_j - np.mean(tau_j)
        tau_hat_j_c = tau_hat_j - np.mean(tau_hat_j)
        corr = float(np.corrcoef(tau_j_c, tau_hat_j_c)[0, 1]) if (np.std(tau_j_c)>1e-9 and np.std(tau_hat_j_c)>1e-9) else np.nan

        print(f"J{j+1}: RMSE={rmse:.4f} Nm  R2={r2:.4f}  corr={corr:.4f}  RMS(τ)={rms(tau_j):.4f}")

        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(q[:, j]);  axs[0].set_ylabel(f"q{j+1} [rad]"); axs[0].grid(True)
        axs[1].plot(dq[:, j]); axs[1].set_ylabel(f"dq{j+1} [rad/s]"); axs[1].grid(True)
        axs[2].plot(tau[:, j], label="τ real")
        axs[2].plot(tau_est_block[:, j], "--", label="τ̂")
        axs[2].set_ylabel(f"τ{j+1} [Nm]")
        axs[2].set_xlabel("muestra")
        axs[2].grid(True)
        axs[2].legend()
        fig.suptitle(f"{tag} — Joint {j+1} (TRAIN)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_TRAIN, f"{tag}_J{j+1}.png"), dpi=130)
        plt.close(fig)

        (start1, end1), (start2, end2) = two_zoom_windows(dq_j_all, N, ZOOM_LEN)
        for (start, end, tag_phase) in [(start1,end1,"PHASE1_RANDOM"), (start2,end2,"PHASE2_SPIRAL")]:
            x_z       = np.arange(start, end)
            q_z       = q[start:end, j]
            dq_z      = dq[start:end, j]
            tau_z     = tau[start:end, j]
            tau_est_z = tau_est_block[start:end, j]

            fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
            axs[0].plot(x_z, q_z);  axs[0].set_ylabel(f"q{j+1} [rad]"); axs[0].grid(True)
            axs[1].plot(x_z, dq_z); axs[1].set_ylabel(f"dq{j+1} [rad/s]"); axs[1].grid(True)
            axs[2].plot(x_z, tau_z, label="τ real")
            axs[2].plot(x_z, tau_est_z, "--", label="τ̂")
            axs[2].set_ylabel(f"τ{j+1} [Nm]"); axs[2].set_xlabel("muestra")
            axs[2].grid(True); axs[2].legend()
            fig.suptitle(f"{tag} — Joint {j+1} ({tag_phase} ZOOM)")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR_TRAIN, f"{tag}_J{j+1}_ZOOM_{tag_phase}.png"), dpi=120)
            plt.close(fig)

eval_block_train_ext(theta_ext,  "FULL_EXT")
eval_block_train_ext(theta_J567, "BLOCK_J5J6J7_EXT")
eval_block_train_ext(theta_J67,  "BLOCK_J6J7_EXT")
eval_block_train_ext(theta_J7,   "BLOCK_J7_EXT")

print(f"\n[OK] Todo guardado en: {OUT_DIR_TRAIN}")

