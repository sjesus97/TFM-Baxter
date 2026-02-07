
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from sklearn.metrics import mean_squared_error, r2_score
import os

# ===============================
# 0) CONFIG
# ===============================
CSV_TEST   = "data_test_phi60_theta140.csv"   # circle+square
THETA_FILE = "theta_step1_wls_ls.npy"         # obtenido en train (con WLS clipeado)
SIGMA_FILE = "w_sigma_step1.npy"              # escalas WLS clipeadas guardadas en train
NUM_JOINTS = 7
NPJ        = 10                                # 7x10 = 70

assert os.path.exists(CSV_TEST),   f"No encuentro {CSV_TEST}"
assert os.path.exists(THETA_FILE), f"No encuentro {THETA_FILE}"
assert os.path.exists(SIGMA_FILE), f"No encuentro {SIGMA_FILE}"

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
# 2) NEWTON–EULER (DH nuevo) → (7,70)
# ===============================
@jit(nopython=True)
def newtonEulerEstimate(angles, velocities, accels):
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
    thetas[1] += half_pi  # convención DH

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
            linearAccs[:, 0]  = np.array([0.0, 0.0, 0.0])  # g entra en W
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
    U55 = w55;  U56 = T6 @ w66;                    U57 = T6 @ T7 @ w77
    U44 = w44;  U45 = T5 @ w55;                    U46 = T5 @ T6 @ w66;               U47 = T5 @ T6 @ T7 @ w77
    U33 = w33;  U34 = T4 @ w44;                    U35 = T4 @ T5 @ w55;               U36 = T4 @ T5 @ T6 @ w66;         U37 = T4 @ T5 @ T6 @ T7 @ w77
    U22 = w22;  U23 = T3 @ w33;                    U24 = T3 @ T4 @ w44;               U25 = T3 @ T4 @ T5 @ w55;         U26 = T3 @ T4 @ T5 @ T6 @ w66;  U27 = T3 @ T4 @ T5 @ T6 @ T7 @ w77
    U11 = w11;  U12 = T2 @ w22;                    U13 = T2 @ T3 @ w33;               U14 = T2 @ T3 @ T4 @ w44;         U15 = T2 @ T3 @ T4 @ T5 @ w55;  U16 = T2 @ T3 @ T4 @ T5 @ T6 @ w66;  U17 = T2 @ T3 @ T4 @ T5 @ T6 @ T7 @ w77

    obs = np.zeros((7, 70))
    last = 5  # componente de momento z en el frame de la junta

    # Triangular por filas (joints)
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

    return obs

# ===============================
# 3) CARGA TEST
# ===============================
print(f"Leyendo test: {CSV_TEST}")
df = pd.read_csv(CSV_TEST)
q   = df[[f"q{j+1}"   for j in range(NUM_JOINTS)]].to_numpy()
dq  = df[[f"dq{j+1}"  for j in range(NUM_JOINTS)]].to_numpy()
ddq = df[[f"ddq{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
tau = df[[f"tau{j+1}" for j in range(NUM_JOINTS)]].to_numpy()
N = q.shape[0]
print("Muestras test:", N)

# ===============================
# 4) MATRIZ Y_test (N*7 x 70) y tau_vec_test (N*7,)
# ===============================
matObs70_test = np.zeros((N*NUM_JOINTS, NUM_JOINTS*NPJ))
tau_vec_test  = np.zeros(N*NUM_JOINTS)

for i in range(N):
    obs = newtonEulerEstimate(q[i], dq[i], ddq[i])  # (7,70)
    base = i*NUM_JOINTS
    for j in range(NUM_JOINTS):
        matObs70_test[base + j, :] = obs[j, :]
        tau_vec_test[base + j]     = tau[i, j]

print("matObs70_test:", matObs70_test.shape, "tau_vec_test:", tau_vec_test.shape)

# ===============================
# 5) CARGA THETA (train) y SIGMA (WLS train, ya clipeadas)
# ===============================
theta = np.load(THETA_FILE)
sigma = np.load(SIGMA_FILE)   # mismas escalas del entrenamiento (clipping aplicado allí)
assert theta.shape[0] == NUM_JOINTS*NPJ, "theta de tamaño inesperado"
assert sigma.shape[0] == NUM_JOINTS, "sigma de tamaño inesperado"
print("Escalas WLS usadas (clipped, del train):", sigma.round(6))

# ===============================
# 6) PREDICCIÓN (sin pesos para comparar en Nm)
# ===============================
tau_est_vec = matObs70_test @ theta
tau_est = tau_est_vec.reshape(N, NUM_JOINTS)

print("\n=== Métricas por joint (VALIDACIÓN: circle+square) ===")
for j in range(NUM_JOINTS):
    mse = mean_squared_error(tau[:, j], tau_est[:, j])
    r2  = r2_score(tau[:, j], tau_est[:, j])
    print(f"Joint {j+1}: MSE={mse:.6f}, R2={r2:.4f}")

# ===============================
# 7) Métricas ponderadas (diagnóstico WLS) — mismas σ que en train
# ===============================
w = np.zeros(N*NUM_JOINTS)
for j in range(NUM_JOINTS):
    idx = np.arange(j, N*NUM_JOINTS, NUM_JOINTS)
    w[idx] = 1.0 / (sigma[j] + 1e-12)
Wsqrt = np.sqrt(w)
A_test_w = matObs70_test * Wsqrt[:, None]
b_test_w = tau_vec_test * Wsqrt
mse_w_global = np.mean((A_test_w @ theta - b_test_w)**2)
print(f"\nMSE ponderado global (WLS, diagnóstico): {mse_w_global:.6f}")

print("\nMSE ponderado por joint (WLS, diagnóstico):")
for j in range(NUM_JOINTS):
    err_j = (tau_est[:, j] - tau[:, j]) / (sigma[j] + 1e-12)
    mse_w_j = np.mean(err_j**2)
    print(f"  J{j+1}: {mse_w_j:.6f}")

# ===============================
# 8) GRÁFICAS τ vs τ̂ por joint
# ===============================
for j in range(NUM_JOINTS):
    plt.figure(figsize=(10, 4))
    plt.plot(tau[:, j], label=f"τ real J{j+1}")
    plt.plot(tau_est[:, j], "--", label=f"τ estimado J{j+1}")
    plt.title(f"Validación φ=60°, θ=140° — circle+square — Joint {j+1}")
    plt.xlabel("muestra"); plt.ylabel("Torque [Nm]")
    plt.grid(True); plt.legend(); plt.tight_layout(); plt.show()

