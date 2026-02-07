import os, json
import numpy as np
import pandas as pd

from data_generator.analytic_model import set_up_world, limb_left  # brazo izq
from data_generator.path_generator import circle_path, square_path
from data_generator.xyz2joint import get_ang_joints
import utils as ut  # preprocess_traj (q,qd,qdd)

# ================ Config (TRAIN HÍBRIDO) ================
DT = 1/500.0
PHI  = 60
THETA= 140

# ---- Paquete J5-focus (lento + radio mayor) ----
PLANES_J5       = ["xy", "xz", "yz"]
RADII_J5        = [0.02, 0.03]
T_LIST_J5       = [2.0, 3.0]
ORIENT_AMPS_J5  = [25.0, 35.0]     
PHASES_J5       = [0.0, 90.0]
REPEAT_J5       = 3

# ---- Paquete J6-focus (rápido + radio pequeño) ----
PLANES_J6       = ["xy"]
RADII_J6        = [0.006, 0.008, 0.01, 0.012]
T_LIST_J6       = [0.8, 1.0, 1.3]
ORIENT_AMPS_J6  = [28.0, 35.0]
PHASES_J6       = [30.0, 90.0, 150]
REPEAT_J6       = 4

SAVE_CSV  = "wrist56_train_basic.csv"
SAVE_META = "wrist56_train_basic_meta.json"
CENTER_CACHE = "center_cache.json"

# ================ Utiles ================
def load_center(cache_path=CENTER_CACHE):
    assert os.path.exists(cache_path), f"No encuentro {cache_path}. Ejecuta antes: python compute_center.py --phi {PHI} --theta {THETA}"
    with open(cache_path, "r") as f:
        obj = json.load(f)
    c = obj.get("center", None)
    assert isinstance(c, list) and len(c) == 3, f"{cache_path} inválido (sin 'center')"
    return np.array(c, dtype=float), obj

def euler_rpy_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)
    w = cy*cp*cr + sy*sp*sr
    x = cy*cp*sr - sy*sp*cr
    y = cy*sp*cr + sy*cp*sr
    z = sy*cp*cr - cy*sp*sr
    return np.array([x,y,z,w], dtype=float)

def stack_and_save(rows, save_csv):
    data = np.vstack(rows)
    cols = [f"q{i+1}" for i in range(7)] \
         + [f"dq{i+1}" for i in range(7)] \
         + [f"ddq{i+1}" for i in range(7)] \
         + [f"tau{i+1}" for i in range(7)]
    pd.DataFrame(data, columns=cols).to_csv(save_csv, index=False)
    print(f"[OK] CSV: {save_csv}  ({data.shape[0]} muestras)")

def segment_q_from_xyz_with_fixed_quat(x, y, z, quat, limb):
    end_pos = np.column_stack([x, y, z])
    pad = 100
    end_pos_ext = np.vstack([end_pos[-pad:], end_pos, end_pos[:pad]])
    q_ext = get_ang_joints(end_pos_ext, end_orientation=quat)
    q, dq, ddq = ut.preprocess_traj(q_ext, DT)
    tau = limb.get_torq_traj(q, dq, ddq)
    return q[pad:-pad], dq[pad:-pad], ddq[pad:-pad], tau[pad:-pad]

def concat_segments(segments):
    q   = np.concatenate([s[0] for s in segments], axis=0)
    dq  = np.concatenate([s[1] for s in segments], axis=0)
    ddq = np.concatenate([s[2] for s in segments], axis=0)
    tau = np.concatenate([s[3] for s in segments], axis=0)
    return q, dq, ddq, tau

def make_segments(x, y, z, amps_deg, phases_deg, limb):
    segs = []
    N = len(x)
    cuts = np.linspace(0, N, num=3+1, dtype=int)  # 3 tramos
    for k in range(3):
        amp_roll  = np.deg2rad(amps_deg[k % len(amps_deg)])
        amp_pitch = np.deg2rad(amps_deg[(k+1) % len(amps_deg)])
        phase_deg = phases_deg[k % len(phases_deg)]
        quat_k = euler_rpy_to_quat(
            roll = amp_roll  * np.sign(np.cos(np.deg2rad(phase_deg))),
            pitch= amp_pitch * np.sign(np.sin(np.deg2rad(phase_deg))),
            yaw  = 0.0
        )
        sl = slice(cuts[k], cuts[k+1])
        segs.append(segment_q_from_xyz_with_fixed_quat(x[sl], y[sl], z[sl], quat_k, limb))
    return concat_segments(segs)

def rms(x): return float(np.sqrt(np.mean(np.square(x))))

def preview_energy(rows):
    M = np.vstack(rows)  # cada fila: [q(7) | dq(7) | ddq(7) | tau(7)]
    dq  = M[:, 7:14]
    ddq = M[:, 14:21]
    tau = M[:, 21:28]
    print(f"[PREVIEW] J5: RMS(dq)={rms(dq[:,4]):.4f}  RMS(ddq)={rms(ddq[:,4]):.4f}  RMS(tau)={rms(tau[:,4]):.4f}")
    print(f"[PREVIEW] J6: RMS(dq)={rms(dq[:,5]):.4f}  RMS(ddq)={rms(ddq[:,5]):.4f}  RMS(tau)={rms(tau[:,5]):.4f}")

# ================ Main ================
def main():
    if os.path.exists(SAVE_CSV): os.remove(SAVE_CSV)

    set_up_world(1.0/DT)
    center, meta_center = load_center()
    print(f"[CENTER] Usando center_cache.json -> {center}  (phi={meta_center.get('phi')}, theta={meta_center.get('theta')})")

    rows = []

    # -------- Paquete J5-focus --------
    print("\n[PAQUETE] J5-focus (lento + radio mayor)")
    for plane in PLANES_J5:
        for r in RADII_J5:
            for T in T_LIST_J5:
                # círculo
                x_c, y_c, z_c = circle_path(r, center, plane=plane, T=T, dt=DT, repeat=REPEAT_J5)
                q, dq, ddq, tau = make_segments(x_c, y_c, z_c, ORIENT_AMPS_J5, PHASES_J5, limb_left)
                rows.append(np.hstack([q, dq, ddq, tau]))
                # cuadrado
                x_s, y_s, z_s = square_path(center, np.deg2rad(PHI), l=2*r, T=T*1.25, dt=DT, repeat=REPEAT_J5)
                q, dq, ddq, tau = make_segments(x_s, y_s, z_s, ORIENT_AMPS_J5, PHASES_J5, limb_left)
                rows.append(np.hstack([q, dq, ddq, tau]))
                print(f"  [J5] plane={plane} r={r:.3f} T={T:.1f}  Ncirc={len(x_c)} Nsq={len(x_s)}")

    # -------- Paquete J6-focus --------
    print("\n[PAQUETE] J6-focus (rápido + radio pequeño)")
    for plane in PLANES_J6:
        for r in RADII_J6:
            for T in T_LIST_J6:
                # círculo
                x_c, y_c, z_c = circle_path(r, center, plane=plane, T=T, dt=DT, repeat=REPEAT_J6)
                q, dq, ddq, tau = make_segments(x_c, y_c, z_c, ORIENT_AMPS_J6, PHASES_J6, limb_left)
                rows.append(np.hstack([q, dq, ddq, tau]))
                # cuadrado (si r=0, uso un lado mínimo para evitar degenerar)
                side = 2*r if r > 0 else 0.01
                x_s, y_s, z_s = square_path(center, np.deg2rad(PHI), l=side, T=T*1.25, dt=DT, repeat=REPEAT_J6)
                q, dq, ddq, tau = make_segments(x_s, y_s, z_s, ORIENT_AMPS_J6, PHASES_J6, limb_left)
                rows.append(np.hstack([q, dq, ddq, tau]))
                print(f"  [J6] plane={plane} r={r:.3f} T={T:.1f}  Ncirc={len(x_c)} Nsq={len(x_s)}")

    # Preview de energía (diagnóstico rápido)
    preview_energy(rows)

    # Guardado
    stack_and_save(rows, SAVE_CSV)
    meta = dict(
        kind="wrist56_train_hybrid", dt=DT, phi=PHI, theta=THETA,
        center=list(map(float, center)),
        center_source="center_cache.json",
        J5=dict(planes=PLANES_J5, radii=RADII_J5, T_list=T_LIST_J5,
                orient_amps_deg=ORIENT_AMPS_J5, phases_deg=PHASES_J5, repeat=REPEAT_J5),
        J6=dict(planes=PLANES_J6, radii=RADII_J6, T_list=T_LIST_J6,
                orient_amps_deg=ORIENT_AMPS_J6, phases_deg=PHASES_J6, repeat=REPEAT_J6),
        file=SAVE_CSV
    )
    with open(SAVE_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] META: {SAVE_META}")

if __name__ == "__main__":
    main()

