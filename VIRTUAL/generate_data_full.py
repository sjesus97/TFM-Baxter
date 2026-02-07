import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from data_generator.trajectory2q_qd_qdd_tau import get_data

# --- Configuración física ---
JOINT_LIMITS = np.array([
    [-1.7016,  1.7016],
    [-2.147,   1.047],
    [-3.0541,  3.0541],
    [-0.05,    2.618],
    [-3.059,   3.059],
    [-1.5708,  2.094],
    [-3.059,   3.059],
])
NUM_JOINTS = 7
DT = 0.002 # 1/500Hz

SAVE_PATH = "data_train_phi75_theta145.csv"
METADATA_PATH = "data_metadata_total.json"
SHOW_DEBUG_PLOTS = False  

def _derivate(arr, dt):
    d_arr = np.zeros_like(arr)
    d_arr[1:-1] = (arr[2:] - arr[:-2]) / (2*dt)
    d_arr[0] = d_arr[1]
    d_arr[-1] = d_arr[-2]
    return d_arr

columns = (
    [f"q{j+1}" for j in range(NUM_JOINTS)] +
    [f"dq{j+1}" for j in range(NUM_JOINTS)] +
    [f"ddq{j+1}" for j in range(NUM_JOINTS)] +
    [f"tau{j+1}" for j in range(NUM_JOINTS)]
)

if os.path.exists(SAVE_PATH):
    os.remove(SAVE_PATH)

# --- Combinaciones ---
comb_exc = [  
    (75, 145),

]

traj_types_saved = []
traj_lengths_saved = []

for phi, theta in comb_exc:
    print(f"\nGenerando (phi={phi}, theta={theta}) ...")
    trajs = get_data(phi, theta, plot_traj=False)
    for traj_type, traj in zip(["random", "spiral"], trajs):
        if traj.shape[1] != 21:
            print(f"Trayectoria inesperada: {traj.shape}")
            continue
        q = traj[:, 0:7]
        dq = traj[:, 7:14]
        tau = traj[:, 14:21]
        ddq = _derivate(dq, DT)
        # Clip a límites físicos
        for j in range(NUM_JOINTS):
            q[:, j] = np.clip(q[:, j], JOINT_LIMITS[j,0], JOINT_LIMITS[j,1])
        df_traj = pd.DataFrame(np.hstack([q, dq, ddq, tau]), columns=columns)
        df_traj.to_csv(SAVE_PATH, mode='a', header=not os.path.exists(SAVE_PATH), index=False)
        traj_types_saved.append(f"{traj_type}_phi{phi}_theta{theta}")
        traj_lengths_saved.append(len(q))
        if SHOW_DEBUG_PLOTS:
            for joint in range(NUM_JOINTS):
                plt.figure(figsize=(14, 4))
                plt.subplot(1,3,1)
                plt.plot(q[:,joint], label="q")
                plt.title(f"q joint {joint+1}")
                plt.grid(True)
                plt.subplot(1,3,2)
                plt.plot(dq[:,joint], label="dq")
                plt.title(f"dq joint {joint+1}")
                plt.grid(True)
                plt.subplot(1,3,3)
                plt.plot(ddq[:,joint], label="ddq")
                plt.title(f"ddq joint {joint+1}")
                plt.grid(True)
                plt.tight_layout()
                plt.show()

metadata = {
    "description": "Dataset completo para identificación dinámica. Incluye combinaciones excitantes validadas y referencia (phi=60, theta=140).",
    "sampling_dt": DT,
    "columns": columns,
    "traj_types": traj_types_saved,
    "traj_lengths": traj_lengths_saved,
    "joint_limits_rad": JOINT_LIMITS.tolist(),
}
with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=4)
print(f"\nDatos guardados en: {SAVE_PATH}")
print(f"Metadatos guardados en: {METADATA_PATH}")

