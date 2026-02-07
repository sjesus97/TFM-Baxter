
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 0) CONFIG
# ===============================
CSV_INPUT   = "data_train_phi75_theta145_art_def_butter_fc15.csv"

# theta global (70 parámetros) y pesos WLS
SAVE_THETA_FULL   = "theta_J14_real_phi75_theta145_filt.npy"
SAVE_WSIGMA       = "w_sigma_J14_real_phi75_theta145_filt.npy"

# thetas específicas de muñeca (acolchadas a 70)
SAVE_THETA_J567   = "theta_J567_real_phi75_theta145_filt.npy"
SAVE_THETA_J67    = "theta_J67_real_phi75_theta145_filt.npy"
SAVE_THETA_J7     = "theta_J7_real_phi75_theta145_filt.npy"

# Directorio de figuras
OUT_DIR_TRAIN = "/home/jesus/TFM_Baxter/PROCESS/FASE_2_BAXTER_REAL/FASE_2.5_NUEVO_FILTRADO/GRAFICAS"
os.makedirs(OUT_DIR_TRAIN, exist_ok=True)

NUM_JOINTS = 7
NUM_PARAMS_PER_JOINT = 10   # sin fricción → 70 en total
EPS = 1e-12

# ===============================
# 1) FUNCIONES AUXILIARES 
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
def s(parameter, joint):
    return np.sin(parameter[joint])

@jit(nopython=True)
def c(parameter, joint):
    return np.cos(parameter[joint])

@jit(nopython=True)
def transMat(link, alphas, thetas):
    # Rotación 3x3 (misma forma que en el script de tu profesor)
    return np.array([
        [c(thetas, link), -s(thetas, link), 0.0],
        [s(thetas, link) * c(alphas, link),  c(thetas, link) * c(alphas, link), -s(alphas, link)],
        [s(thetas, link) * s(alphas, link),  c(thetas, link) * s(alphas, link),  c(alphas, link)],
    ])

# ===============================
# 2) NEWTON–EULER con DH CORREGIDO
# ===============================
@jit(nopython=True)
def newtonEulerEstimate(angles, velocities, accels):
    half_pi = np.pi / 2.0

   
    alphas = np.array([0.0, -half_pi, half_pi, -half_pi, half_pi, -half_pi, half_pi])
    z = np.array([0.0, 0.0, 1.0])

    # pArray
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

    grav = np.array([0.0, 0.0, -9.8])  

    # Forward Newton–EULER
    for link in range(7):
        rotMat = transMat(link, alphas, thetas)

        if link == 0:
            angularVels[:, 0] = velocities[0] * z
            angularAccs[:, 0] = accels[0] * z
            # la gravedad se inyecta en W, no en linearAccs
            linearAccs[:, 0]  = np.array([0.0, 0.0, 0.0])
        else:
            ang_prev = rotMat.T @ angularVels[:, link-1]
            angularVels[:, link] = ang_prev + velocities[link] * z

            angularAccs[:, link] = rotMat.T @ angularAccs[:, link-1] \
                                 + np.cross(ang_prev,
                                            velocities[link] * z + accels[link] * z)

            linearAccs[:, link]  = rotMat.T @ (
                                        linearAccs[:, link-1]
                                      + np.cross(angularAccs[:, link-1], pArray[:, link])
                                      + np.cross(angularVels[:, link-1],
                                                 np.cross(angularVels[:, link-1], pArray[:, link]))
                                   )

        # Matriz W (6x10) para el link actual
        W_up = np.concatenate((
            (linearAccs[:, link] - grav).reshape(3, 1),
            crossMat(angularAccs[:, link]) +
            crossMat(angularVels[:, link]) @ crossMat(angularVels[:, link]),
            np.zeros((3, 6))
        ), axis=1)

        W_lo = np.concatenate((
            np.zeros((3, 1)),
            crossMat(grav - linearAccs[:, link]),
            dotMat(angularAccs[:, link]) +
            crossMat(angularVels[:, link]) @ dotMat(angularVels[:, link])
        ), axis=1)

        w_j_j[:, :, link] = np.concatenate((W_up, W_lo), axis=0)

        # Matriz de transformación 6x6 para el esquema triangular
        T_up = np.concatenate((rotMat, np.zeros((3, 3))), axis=1)
        T_lo = np.concatenate((crossMat(pArray[:, link]) @ rotMat, rotMat), axis=1)
        TMatrices[:, :, link] = np.concatenate((T_up, T_lo), axis=0)

    # Alias y construcción triangular (Atkeson)
    w11 = w_j_j[:, :, 0]
    w22 = w_j_j[:, :, 1]
    w33 = w_j_j[:, :, 2]
    w44 = w_j_j[:, :, 3]
    w55 = w_j_j[:, :, 4]
    w66 = w_j_j[:, :, 5]
    w77 = w_j_j[:, :, 6]

    T2 = TMatrices[:, :, 1]
    T3 = TMatrices[:, :, 2]
    T4 = TMatrices[:, :, 3]
    T5 = TMatrices[:, :, 4]
    T6 = TMatrices[:, :, 5]
    T7 = TMatrices[:, :, 6]

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

    # Observación final (7x70)
    obs = np.zeros((7, 70))

    
    li = -1

    # Joint 1
    obs[0,   0: 10] = U11[li, :]
    obs[0,  10: 20] = U12[li, :]
    obs[0,  20: 30] = U13[li, :]
    obs[0,  30: 40] = U14[li, :]
    obs[0,  40: 50] = U15[li, :]
    obs[0,  50: 60] = U16[li, :]
    obs[0,  60: 70] = U17[li, :]
    # Joint 2
    obs[1,  10: 20] = U22[li, :]
    obs[1,  20: 30] = U23[li, :]
    obs[1,  30: 40] = U24[li, :]
    obs[1,  40: 50] = U25[li, :]
    obs[1,  50: 60] = U26[li, :]
    obs[1,  60: 70] = U27[li, :]
    # Joint 3
    obs[2,  20: 30] = U33[li, :]
    obs[2,  30: 40] = U34[li, :]
    obs[2,  40: 50] = U35[li, :]
    obs[2,  50: 60] = U36[li, :]
    obs[2,  60: 70] = U37[li, :]
    # Joint 4
    obs[3,  30: 40] = U44[li, :]
    obs[3,  40: 50] = U45[li, :]
    obs[3,  50: 60] = U46[li, :]
    obs[3,  60: 70] = U47[li, :]
    # Joint 5
    obs[4,  40: 50] = U55[li, :]
    obs[4,  50: 60] = U56[li, :]
    obs[4,  60: 70] = U57[li, :]
    # Joint 6
    obs[5,  50: 60] = U66[li, :]
    obs[5,  60: 70] = U67[li, :]
    # Joint 7
    obs[6,  60: 70] = U77[li, :]

    return obs  # (7,70)

# ---------- utilidades métricas ----------
def rms(x):
    return float(np.sqrt(np.mean(x**2)))

def row_idx_for_joint(j, N, n_joints=7):
    return np.arange(j, N*n_joints, n_joints)

def two_zoom_windows(dq_j, N, zoom_len):
    """
    Devuelve dos ventanas de zoom (fase1, fase2) para un joint:
    - fase1 ~ primera mitad del dataset (random)
    - fase2 ~ segunda mitad (spiral)
    """
    mid = N // 2

    def find_window(start_phase, end_phase):
        dq_seg = dq_j[start_phase:end_phase]
        max_abs = np.max(np.abs(dq_seg))
        if max_abs > 0:
            thr = 0.2 * max_abs
            active = np.where(np.abs(dq_seg) > thr)[0]
        else:
            active = np.zeros(0, dtype=np.int64)

        if active.size > 0:
            start = start_phase + active[0]
        else:
            start = start_phase
        end = min(start + zoom_len, end_phase)
        return start, end

    z1 = find_window(0, mid)
    z2 = find_window(mid, N)
    return z1, z2

# ===============================
# 3) CARGA DE DATOS
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
# 3.1 DIAGNÓSTICO PREVIO EN τ
# ===============================
print("\n================= DIAGNÓSTICO PREVIO EN τ =================")
mean_tau = tau.mean(axis=0)
std_tau  = tau.std(axis=0)
rms_tau  = np.sqrt(np.mean(tau**2, axis=0))

for j in range(NUM_JOINTS):
    print(f"Joint {j+1}: mean={mean_tau[j]:+.4f}, std={std_tau[j]:.4f}, RMS={rms_tau[j]:.4f}")

print("\nCorrelaciones físicas q/dq/ddq con τ (J1 y J2):")
for j in [0, 1]:
    corr_q   = np.corrcoef(q[:, j],   tau[:, j])[0, 1]
    corr_dq  = np.corrcoef(dq[:, j],  tau[:, j])[0, 1]
    corr_ddq = np.corrcoef(ddq[:, j], tau[:, j])[0, 1]
    print(f"Joint {j+1}: corr(q,τ)={corr_q:+.3f}, corr(dq,τ)={corr_dq:+.3f}, corr(ddq,τ)={corr_ddq:+.3f}")

print("\nTest de cambio de signo en τ1 y τ2:")
tau1_inv = -tau[:, 0]
tau2_inv = -tau[:, 1]
corr_tau1_inv = np.corrcoef(tau1_inv, tau[:, 0])[0, 1]
corr_tau2_inv = np.corrcoef(tau2_inv, tau[:, 1])[0, 1]
print(f"corr(τ1, -τ1) = {corr_tau1_inv:.3f} (≈ -1)")
print(f"corr(τ2, -τ2) = {corr_tau2_inv:.3f} (≈ -1)")

print("\nTest rápido de correlación cruzada τ1 con τ2:")
corr_12 = np.corrcoef(tau[:, 0], tau[:, 1])[0, 1]
print(f"corr(τ1, τ2) = {corr_12:.3f}")
print("=======================================================\n")

# ===============================
# 4) CONSTRUCCIÓN Y (N*7 x 70) y τ_vec (N*7,)
# ===============================
matObs70 = np.zeros((N*NUM_JOINTS, NUM_JOINTS*NUM_PARAMS_PER_JOINT))
tau_vec  = np.zeros(N*NUM_JOINTS)

for i in range(N):
    obs = newtonEulerEstimate(q[i], dq[i], ddq[i])  # (7,70)
    base = i * NUM_JOINTS
    for j in range(NUM_JOINTS):
        matObs70[base + j, :] = obs[j, :]
        tau_vec[base + j]     = tau[i, j]

print("matObs70:", matObs70.shape, "tau_vec:", tau_vec.shape)

# ===============================
# 4.1 DIAGNÓSTICO DE EXCITACIÓN EN Y (J1 vs J2)
# ===============================
print("\n============= DIAGNÓSTICO DE EXCITACIÓN EN Y =================")
cols_J1 = np.arange(0, 10)
cols_J2 = np.arange(10, 20)

rms_Y_J1 = np.sqrt(np.mean(matObs70[:, cols_J1]**2, axis=0))
rms_Y_J2 = np.sqrt(np.mean(matObs70[:, cols_J2]**2, axis=0))

print("RMS columnas Y (Joint 1):")
print(np.round(rms_Y_J1, 6))

print("\nRMS columnas Y (Joint 2):")
print(np.round(rms_Y_J2, 6))

ratio_exc = (np.mean(rms_Y_J2) / (np.mean(rms_Y_J1) + 1e-12))
print(f"\nRatio excitación J2/J1 = {ratio_exc:.3f}")
print("=======================================================\n")

# ===============================
# 5) WLS por articulación (cada joint pesa lo mismo)
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
A = matObs70 * Wsqrt[:, None]
b = tau_vec * Wsqrt

np.save(SAVE_WSIGMA, sigma)
print("Escalas por joint (RMS):", sigma.round(6))

# ===============================
# 6) REGRESIÓN GLOBAL (LS + WLS) — modelo completo J1–J7
# ===============================
theta_full, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
np.save(SAVE_THETA_FULL, theta_full)
print(f"theta_full guardado en {SAVE_THETA_FULL}")
print(f"Rango(A) = {rank} / 70")

# ===============================
# 7) PREDICCIÓN GLOBAL
# ===============================
tau_est_vec = matObs70 @ theta_full
tau_est = tau_est_vec.reshape(N, NUM_JOINTS)

# ----- Offset τ_real - τ_est -----
print("\n================= TEST DE OFFSET τ REAL - τ EST =================")
for j in range(2):  # centrado en J1 y J2
    offset = float(np.mean(tau[:, j] - tau_est[:, j]))
    print(f"Joint {j+1}: offset real-estimado = {offset:+.4f} Nm")
print("=======================================================\n")

print("\n=== Métricas por joint (LS + WLS, datos FILTRADOS — modelo global) ===")
ZOOM_LEN = min(2000, N)   # antes 4000 → ahora el doble de zoom

for j in range(NUM_JOINTS):
    tau_j = tau[:, j]
    tau_hat_j = tau_est[:, j]
    dq_j = dq[:, j]

    # Métricas estándar
    mse = mean_squared_error(tau_j, tau_hat_j)
    rmse = np.sqrt(mse)
    rms_tau_j = rms(tau_j)
    rmse_rel = 100.0 * rmse / (rms_tau_j + EPS)
    r2 = r2_score(tau_j, tau_hat_j)

    # Métricas centradas (quitamos bias) + correlación
    tau_j_c = tau_j - np.mean(tau_j)
    tau_hat_j_c = tau_hat_j - np.mean(tau_hat_j)

    mse_c = mean_squared_error(tau_j_c, tau_hat_j_c)
    rmse_c = np.sqrt(mse_c)
    var_tau_c = np.var(tau_j_c) + EPS
    r2_centered = 1.0 - mse_c / var_tau_c

    if np.std(tau_j_c) > 1e-9 and np.std(tau_hat_j_c) > 1e-9:
        corr = float(np.corrcoef(tau_j_c, tau_hat_j_c)[0, 1])
    else:
        corr = np.nan

    # R² activo según dq
    max_abs_dq = np.max(np.abs(dq_j))
    if max_abs_dq > 0:
        thr = 0.2 * max_abs_dq
        mask_act = np.abs(dq_j) > thr
        if np.any(mask_act):
            r2_act = r2_score(tau_j[mask_act], tau_hat_j[mask_act])
        else:
            r2_act = np.nan
    else:
        r2_act = np.nan

    print(
        f"Joint {j+1}: "
        f"MSE={mse:.6f}, RMSE={rmse:.4f} Nm, RMSE_rel={rmse_rel:.2f}%, "
        f"R2={r2:.4f}, R2_activo={r2_act:.4f}, "
        f"RMSE_c={rmse_c:.4f} Nm, R2_centrado={r2_centered:.4f}, "
        f"corr={corr:.4f} "
        f"(RMS(τ)={rms_tau_j:.4f} Nm)"
    )

    # --------- GRÁFICA GLOBAL τ vs τ̂ ----------
    plt.figure(figsize=(10, 4))
    plt.plot(tau_j, label=f"τ real J{j+1}")
    plt.plot(tau_hat_j, "--", label=f"τ estimado J{j+1}")
    plt.title(f"Joint {j+1} — modelo global (train filtrado)")
    plt.xlabel("muestra")
    plt.ylabel("Torque [Nm]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_TRAIN, f"GLOBAL_J{j+1}.png"), dpi=130)
    plt.close()

    # --------- DOS ZOOMS POR JOINT (FASE RANDOM / FASE SPIRAL) ----------
    (start1, end1), (start2, end2) = two_zoom_windows(dq_j, N, ZOOM_LEN)

    # ZOOM 1: fase 1 (random, primera mitad)
    for zoom_id, (start, end, tag_phase) in enumerate(
        [(start1, end1, "PHASE1_RANDOM"), (start2, end2, "PHASE2_SPIRAL")]
    ):
        x_z       = np.arange(start, end)
        q_z       = q[start:end, j]
        dq_z      = dq[start:end, j]
        tau_z     = tau[start:end, j]
        tau_est_z = tau_est[start:end, j]

        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(x_z, q_z)
        axs[0].set_ylabel(f"q{j+1} [rad]")
        axs[0].grid(True)

        axs[1].plot(x_z, dq_z)
        axs[1].set_ylabel(f"dq{j+1} [rad/s]")
        axs[1].grid(True)

        axs[2].plot(x_z, tau_z, label="τ real")
        axs[2].plot(x_z, tau_est_z, "--", label="τ̂")
        axs[2].set_ylabel(f"τ{j+1} [Nm]")
        axs[2].set_xlabel("muestra")
        axs[2].grid(True)
        axs[2].legend()

        fig.suptitle(f"GLOBAL_WLS — Joint {j+1} ({tag_phase} ZOOM)")
        plt.tight_layout()
        fname = f"GLOBAL_J{j+1}_ZOOM_{tag_phase}.png"
        plt.savefig(os.path.join(OUT_DIR_TRAIN, fname), dpi=120)
        plt.close(fig)



# ============================================================
# 7.5 MODELO REDUCIDO PARA J1 (inercial efectivo + fricción + offset)
# ============================================================

print("\n=========== MODELO REDUCIDO PARA J1 (θ_eff + k + b + d) ===========")

# --- 1) Seleccionar la única columna viva de Y_J1 ---
cols_J1 = np.arange(0, 10)
YJ1_block = matObs70[row_idx_for_joint(0, N), :][:, cols_J1]

# Identificar columna viva (la que no es cero)
col_rms = np.sqrt(np.mean(YJ1_block**2, axis=0))
idx_live = np.argmax(col_rms)   # única columna excitada
c_t = YJ1_block[:, idx_live]    # señal c(t)

# --- 2) Variables de fricción ---
dq1 = dq[:, 0]
sign_dq1 = np.sign(dq1)
tau1 = tau[:, 0]

# --- 3) Construir regresor reducido ---
# [ c(t), dq1(t), sign(dq1), 1 ]
X_red = np.column_stack([c_t, dq1, sign_dq1, np.ones_like(c_t)])

# --- 4) Resolver LS reducido ---
theta_red, res, rk, _ = np.linalg.lstsq(X_red, tau1, rcond=None)

theta_eff = theta_red[0]
k_fric    = theta_red[1]
b_coul    = theta_red[2]
d_offset  = theta_red[3]

print("\nParámetros estimados modelo reducido J1:")
print(f"θ_eff   = {theta_eff:+.5f}")
print(f"k       = {k_fric:+.5f}")
print(f"b       = {b_coul:+.5f}")
print(f"d       = {d_offset:+.5f}")

# --- 5) Predicción ---
tau1_red = X_red @ theta_red

# --- 6) Métricas ---
mse_red  = mean_squared_error(tau1, tau1_red)
rmse_red = np.sqrt(mse_red)
r2_red   = r2_score(tau1, tau1_red)

print("\nMétricas modelo reducido J1:")
print(f" RMSE      = {rmse_red:.4f} Nm")
print(f" R²        = {r2_red:.4f}")

# --- 7) Plot ---
plt.figure(figsize=(10,5))
plt.plot(tau1, label="τ1 real")
plt.plot(tau1_red, '--', label="τ1 estimado (reducido)")
plt.title("MODELO REDUCIDO J1 — τ vs τ̂")
plt.xlabel("muestra")
plt.ylabel("Torque [Nm]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR_TRAIN, "REDUCED_J1.png"), dpi=140)
plt.close()

# --- 8) DOS ZOOMS (FASE RANDOM / FASE SPIRAL), COMO EN EL GLOBAL ---
# Usamos la misma lógica de ventanas basada en dq1
(start1, end1), (start2, end2) = two_zoom_windows(dq1, N, ZOOM_LEN)

for (start, end, tag_phase) in [
    (start1, end1, "PHASE1_RANDOM"),
    (start2, end2, "PHASE2_SPIRAL"),
]:
    x_z        = np.arange(start, end)
    q1_z       = q[start:end, 0]
    dq1_z      = dq[start:end, 0]
    tau1_z     = tau[start:end, 0]
    tau1_red_z = tau1_red[start:end]

    fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    # q1
    axs[0].plot(x_z, q1_z)
    axs[0].set_ylabel("q1 [rad]")
    axs[0].grid(True)

    # dq1
    axs[1].plot(x_z, dq1_z)
    axs[1].set_ylabel("dq1 [rad/s]")
    axs[1].grid(True)

    # τ1 real vs τ̂1 reducido
    axs[2].plot(x_z, tau1_z, label="τ1 real")
    axs[2].plot(x_z, tau1_red_z, "--", label="τ̂1 reducido")
    axs[2].set_ylabel("τ1 [Nm]")
    axs[2].set_xlabel("muestra")
    axs[2].grid(True)
    axs[2].legend()

    fig.suptitle(f"MODELO REDUCIDO J1 — {tag_phase} ZOOM")
    plt.tight_layout()
    fname = f"REDUCED_J1_ZOOM_{tag_phase}.png"
    plt.savefig(os.path.join(OUT_DIR_TRAIN, fname), dpi=130)
    plt.close(fig)

# ===============================
# 8) BLOQUES MUÑECA (J5, J6, J7) — desacople sobre mismos datos
# ===============================
COLS_J7   = slice(60, 70)   # 10 parámetros
COLS_J67  = slice(50, 70)   # 20 parámetros
COLS_J567 = slice(40, 70)   # 30 parámetros

ROWS_J7   = row_idx_for_joint(6, N, NUM_JOINTS)
ROWS_J6   = row_idx_for_joint(5, N, NUM_JOINTS)
ROWS_J5   = row_idx_for_joint(4, N, NUM_JOINTS)
ROWS_J67  = np.sort(np.concatenate([ROWS_J6, ROWS_J7]))
ROWS_J567 = np.sort(np.concatenate([ROWS_J5, ROWS_J6, ROWS_J7]))

def fit_block_padded(rows, cols):
    """
    Resuelve el bloque (J7, J6–J7, J5–J6–J7) sobre las filas de esos joints,
    y devuelve θ acolchado (70,), rank, svals, cond.
    """
    A_block = matObs70[rows, cols]
    b_block = tau_vec[rows]

    # WLS local fila a fila según joint al que pertenece
    sigma_row = np.empty(rows.shape[0])
    for k, r in enumerate(rows):
        sigma_row[k] = sigma[r % NUM_JOINTS]
    Wsqrt_loc = 1.0 / np.sqrt(sigma_row + 1e-12)
    A_w = A_block * Wsqrt_loc[:, None]
    b_w = b_block * Wsqrt_loc

    # normalización de columnas
    col_rms = np.sqrt(np.mean(A_w**2, axis=0)) + 1e-12
    A_wn = A_w / col_rms[None, :]

    theta_local_n, residuals, rank, svals = np.linalg.lstsq(A_wn, b_w, rcond=None)
    theta_local = theta_local_n / col_rms

    # acolchado a 70
    theta_pad = np.zeros(70)
    theta_pad[cols] = theta_local

    # condición del bloque SIN pesos (diagnóstico)
    try:
        S = np.linalg.svd(A_block, full_matrices=False)[1]
        cond = (S.max() / S.min()) if np.all(S > 1e-12) else np.inf
    except Exception:
        cond = np.nan

    return theta_pad, rank, svals, cond

theta_J7,   rank_J7,   s_J7,   cond_J7   = fit_block_padded(ROWS_J7,   COLS_J7)
theta_J67,  rank_J67,  s_J67,  cond_J67  = fit_block_padded(ROWS_J67,  COLS_J67)
theta_J567, rank_J567, s_J567, cond_J567 = fit_block_padded(ROWS_J567, COLS_J567)

np.save(SAVE_THETA_J7,   theta_J7)
np.save(SAVE_THETA_J67,  theta_J67)
np.save(SAVE_THETA_J567, theta_J567)

print(f"\n[OK] θ_J7   → {SAVE_THETA_J7}     rank={rank_J7}   cond={cond_J7:.3e}")
print(f"[OK] θ_J67  → {SAVE_THETA_J67}   rank={rank_J67}  cond={cond_J67:.3e}")
print(f"[OK] θ_J567 → {SAVE_THETA_J567} rank={rank_J567} cond={cond_J567:.3e}")

# ===============================
# 9) Métricas TRAIN para los modelos de muñeca (sólo J5–J7)
# ===============================
def eval_block_train(theta_vec, tag):
    tau_est_vec_block = matObs70 @ theta_vec
    tau_est_block = tau_est_vec_block.reshape(N, NUM_JOINTS)

    print(f"\n=== Métricas por joint — {tag} (TRAIN filtrado, sólo mostramos J5–J7) ===")
    for j in [4, 5, 6]:
        tau_j = tau[:, j]
        tau_hat_j = tau_est_block[:, j]

        mse = mean_squared_error(tau_j, tau_hat_j)
        r2  = r2_score(tau_j, tau_est_block[:, j])
        print(f"J{j+1}: MSE={mse:.6f}  R2={r2:.4f}  RMS(τ)={rms(tau_j):.4f}")

        # Gráfica rápida de τ vs τ̂ (toda la señal)
        fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        axs[0].plot(q[:, j]);  axs[0].set_ylabel(f"q{j+1} [rad]"); axs[0].grid(True)
        axs[1].plot(dq[:, j]); axs[1].set_ylabel(f"dq{j+1} [rad/s]"); axs[1].grid(True)
        axs[2].plot(tau[:, j], label="τ real")
        axs[2].plot(tau_est_block[:, j], "--", label="τ̂")
        axs[2].set_ylabel(f"τ{j+1} [Nm]")
        axs[2].set_xlabel("muestra")
        axs[2].grid(True)
        axs[2].legend()
        fig.suptitle(f"{tag} — Joint {j+1} (TRAIN filtrado)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR_TRAIN, f"{tag}_J{j+1}.png"), dpi=130)
        plt.close(fig)

        # --------- DOS ZOOMS POR JOINT TAMBIÉN PARA BLOQUES DE MUÑECA ----------
        dq_j_all = dq[:, j]
        (start1, end1), (start2, end2) = two_zoom_windows(dq_j_all, N, ZOOM_LEN)

        for (start, end, tag_phase) in [
            (start1, end1, "PHASE1_RANDOM"),
            (start2, end2, "PHASE2_SPIRAL")
        ]:
            x_z       = np.arange(start, end)
            q_z       = q[start:end, j]
            dq_z      = dq[start:end, j]
            tau_z     = tau[start:end, j]
            tau_est_z = tau_est_block[start:end, j]

            fig, axs = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
            axs[0].plot(x_z, q_z)
            axs[0].set_ylabel(f"q{j+1} [rad]")
            axs[0].grid(True)

            axs[1].plot(x_z, dq_z)
            axs[1].set_ylabel(f"dq{j+1} [rad/s]")
            axs[1].grid(True)

            axs[2].plot(x_z, tau_z, label="τ real")
            axs[2].plot(x_z, tau_est_z, "--", label="τ̂")
            axs[2].set_ylabel(f"τ{j+1} [Nm]")
            axs[2].set_xlabel("muestra")
            axs[2].grid(True)
            axs[2].legend()

            fig.suptitle(f"{tag} — Joint {j+1} ({tag_phase} ZOOM)")
            plt.tight_layout()
            fname = f"{tag}_J{j+1}_ZOOM_{tag_phase}.png"
            plt.savefig(os.path.join(OUT_DIR_TRAIN, fname), dpi=120)
            plt.close(fig)

eval_block_train(theta_full,  "FULL_WLS")
eval_block_train(theta_J567,  "BLOCK_J5J6J7")
eval_block_train(theta_J67,   "BLOCK_J6J7")
eval_block_train(theta_J7,    "BLOCK_J7")

print(f"\n[OK] Resultados de TRAIN (global + muñeca) guardados en: {OUT_DIR_TRAIN}")

