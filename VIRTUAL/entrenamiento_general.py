import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score

# ===============================
# 0) CONFIG
# ===============================
CSV_INPUT = "data_phi75_theta145.csv"   
SAVE_THETA = "theta_step1_wls_ls.npy"
SAVE_WSIGMA = "w_sigma_step1.npy"
NUM_JOINTS = 7
NUM_PARAMS_PER_JOINT = 10 

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
def s(parameter, joint): return np.sin(parameter[joint])

@jit(nopython=True)
def c(parameter, joint): return np.cos(parameter[joint])

@jit(nopython=True)
def transMat(link, alphas, thetas):
    # Rotación 3x3
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
    half_pi = np.pi / 2
    alphas = np.array([0.0, -half_pi, half_pi, -half_pi, half_pi, -half_pi, half_pi])
    z = np.array([0.0, 0.0, 1.0])
    pArray = np.array([  # (3,7)
        [0.0,      0.0,      0.0     ],
        [0.069,    0.0,      0.27035 ],
        [0.0,     -0.102,    0.0     ],
        [0.069,    0.0,      0.26242 ],
        [0.0,     -0.10359,  0.0     ],
        [0.01,     0.0,      0.2707  ],
        [0.0,     -0.115975, 0.0     ]
    ]).T

    thetas = angles.copy()
    thetas[1] += half_pi  # convención DH 

    angularVels = np.zeros((3, 7))
    angularAccs = np.zeros((3, 7))
    linearAccs  = np.zeros((3, 7))

    w_j_j = np.zeros((6, 10, 7))   # 10 params inerciales/junta
    TMatrices = np.zeros((6, 6, 7))
    grav = np.array([0.0, 0.0, -9.81])

    for link in range(7):
        rotMat = transMat(link, alphas, thetas)
        if link == 0:
            angularVels[:, 0] = velocities[0] * z
            angularAccs[:, 0] = accels[0] * z
            linearAccs[:, 0]  = np.array([0.0, 0.0, 0.0])  # metemos g en W
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

    # Alias y triangular (idéntico a tu profe)
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
    last = 5  # fila 6 (momento z)

    # Joint 1
    obs[0,   0: 10] = U11[last, :]
    obs[0,  10: 20] = U12[last, :]
    obs[0,  20: 30] = U13[last, :]
    obs[0,  30: 40] = U14[last, :]
    obs[0,  40: 50] = U15[last, :]
    obs[0,  50: 60] = U16[last, :]
    obs[0,  60: 70] = U17[last, :]
    # Joint 2
    obs[1,  10: 20] = U22[last, :]
    obs[1,  20: 30] = U23[last, :]
    obs[1,  30: 40] = U24[last, :]
    obs[1,  40: 50] = U25[last, :]
    obs[1,  50: 60] = U26[last, :]
    obs[1,  60: 70] = U27[last, :]
    # Joint 3
    obs[2,  20: 30] = U33[last, :]
    obs[2,  30: 40] = U34[last, :]
    obs[2,  40: 50] = U35[last, :]
    obs[2,  50: 60] = U36[last, :]
    obs[2,  60: 70] = U37[last, :]
    # Joint 4
    obs[3,  30: 40] = U44[last, :]
    obs[3,  40: 50] = U45[last, :]
    obs[3,  50: 60] = U46[last, :]
    obs[3,  60: 70] = U47[last, :]
    # Joint 5
    obs[4,  40: 50] = U55[last, :]
    obs[4,  50: 60] = U56[last, :]
    obs[4,  60: 70] = U57[last, :]
    # Joint 6
    obs[5,  50: 60] = U66[last, :]
    obs[5,  60: 70] = U67[last, :]
    # Joint 7
    obs[6,  60: 70] = U77[last, :]

    return obs  # (7,70)

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
# 4) CONSTRUCCIÓN Y (N*7 x 70) y τ_vec (N*7,)
# ===============================
matObs70 = np.zeros((N*NUM_JOINTS, NUM_JOINTS*NUM_PARAMS_PER_JOINT))
tau_vec  = np.zeros(N*NUM_JOINTS)

for i in range(N):
    obs = newtonEulerEstimate(q[i], dq[i], ddq[i])  # (7,70)
    base = i*NUM_JOINTS
    for j in range(NUM_JOINTS):
        matObs70[base + j, :] = obs[j, :]
        tau_vec[base + j]     = tau[i, j]

print("matObs70:", matObs70.shape, "tau_vec:", tau_vec.shape)

# ===============================
# 5) WLS por articulación
# ===============================
sigma = np.zeros(NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = np.arange(j, N*NUM_JOINTS, NUM_JOINTS)
    # Escala por joint: RMS robusto
    sigma[j] = np.sqrt(np.mean(tau_vec[idx]**2)) + 1e-12

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
# 6) REGRESIÓN 
# ===============================
theta, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
np.save(SAVE_THETA, theta)
print(f"theta guardado en {SAVE_THETA}")
print(f"Rango(A) = {rank} / 70")

# ===============================
# 7) PREDICCIÓN y MÉTRICAS
# ===============================
tau_est_vec = matObs70 @ theta
tau_est = tau_est_vec.reshape(N, NUM_JOINTS)

print("\n=== Métricas por joint (LS + WLS) ===")
for j in range(NUM_JOINTS):
    mse = mean_squared_error(tau[:, j], tau_est[:, j])
    r2  = r2_score(tau[:, j], tau_est[:, j])
    print(f"Joint {j+1}: MSE={mse:.6f}, R2={r2:.4f}")

# ===============================
# 8) GRÁFICAS τ vs τ̂ por joint
# ===============================
for j in range(NUM_JOINTS):
    plt.figure(figsize=(10, 4))
    plt.plot(tau[:, j], label=f"τ real J{j+1}")
    plt.plot(tau_est[:, j], "--", label=f"τ estimado J{j+1}")
    plt.title(f"Joint {j+1} — LS + WLS")
    plt.xlabel("muestra"); plt.ylabel("Torque [Nm]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

