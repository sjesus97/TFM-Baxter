import os, json
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline  # splines cúbicos

from data_generator.analytic_model import set_up_world, limb_left  # brazo izq
from data_generator.xyz2joint import get_ang_joints
import utils as ut  # preprocess_traj (q,qd,qdd)

# ================ Config (VAL ORIENTACIÓN MUÑECA) ================
DT = 1/500.0
PHI  = 60   # sólo meta
THETA= 140
CENTER_CACHE = "center_cache.json"

# VALIDACIÓN: misma escala que TRAIN pero con menos repeticiones / T distintos
ORIENT_AMPS_J5  = [35.0, 55.0]
PHASES_J5       = [0.0, 90.0, 180.0]
T_LIST_J5       = [0.9, 1.6]
REPEAT_J5       = 3
N_SEG_J5        = 8

ORIENT_AMPS_J6  = [30.0, 45.0]
PHASES_J6       = [0.0, 90.0, 180.0]
T_LIST_J6       = [0.9, 1.4]
REPEAT_J6       = 3
N_SEG_J6        = 6


SAVE_CSV  = "wrist56_val_like_train.csv"
SAVE_META = "wrist56_val_like_train_meta.json"

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

def constant_path(center, T, dt, repeat):
    N_seg = int(np.round(T / dt))
    N = N_seg * repeat
    x = np.full(N, center[0])
    y = np.full(N, center[1])
    z = np.full(N, center[2])
    return x, y, z

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

    q_f, dq_f, ddq_f = ut.preprocess_traj(q_ext, DT)
    tau_f = limb.get_torq_traj(q_f, dq_f, ddq_f)

    return q_f[pad:-pad], dq_f[pad:-pad], ddq_f[pad:-pad], tau_f[pad:-pad]

def concat_segments(segments):
    q   = np.concatenate([s[0] for s in segments], axis=0)
    dq  = np.concatenate([s[1] for s in segments], axis=0)
    ddq = np.concatenate([s[2] for s in segments], axis=0)
    tau = np.concatenate([s[3] for s in segments], axis=0)
    return q, dq, ddq, tau

def smooth_wrist_between_segments(q_cat, cuts, dt, joints=(4,5)):
    """
    Splines cúbicos entre segmentos para J5/J6.
    cuts: índices de corte (0..N) en la trayectoria concatenada.
    """
    N = q_cat.shape[0]
    t_all = np.arange(N) * dt

    q_s = q_cat.copy()

    # tiempos "nudo" = centro de cada segmento
    t_knots = []
    q_knots = {j: [] for j in joints}

    for k in range(len(cuts)-1):
        a, b = cuts[k], cuts[k+1]
        if b <= a:
            continue
        t_mid = 0.5*(a + b) * dt
        t_knots.append(t_mid)
        for j in joints:
            q_knots[j].append(np.mean(q_cat[a:b, j]))

    t_knots = np.array(t_knots)
    for j in joints:
        qk = np.array(q_knots[j])
        cs = CubicSpline(t_knots, qk, bc_type='natural')
        q_s[:, j] = cs(t_all)

    return q_s

def extra_smooth_q(q, joints=(4,5), win=11):
    """
    Suavizado extra (media móvil) para J5/J6.
    """
    N, nJ = q.shape
    q_f = q.copy()
    half = win // 2
    kernel = np.ones(win, dtype=float) / win

    for j in joints:
        col = q[:, j]
        pad_left  = col[1:half+1][::-1]
        pad_right = col[-half-1:-1][::-1]
        col_pad = np.concatenate([pad_left, col, pad_right])
        col_s = np.convolve(col_pad, kernel, mode='same')
        q_f[:, j] = col_s[half:-half]

    return q_f

def make_segments_fixed_pos(center, T, amps_deg, phases_deg, repeat, n_segments, limb):
    """
    Igual que en TRAIN pero versión VAL:
    - punto fijo.
    - diferentes T, repeat y n_segments.
    - suavizado J5/J6 tras concatenar segmentos.
    """
    x, y, z = constant_path(center, T, DT, repeat)
    segs = []
    N = len(x)
    cuts = np.linspace(0, N, num=n_segments+1, dtype=int)

    for k in range(n_segments):
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

    # Trayectoria original concatenada
    q_cat, dq_cat, ddq_cat, tau_cat = concat_segments(segs)

    # Suavizado de muñeca entre segmentos
    q_s = smooth_wrist_between_segments(q_cat, cuts, DT, joints=(4,5))

    # Suavizado extra tipo low-pass
    q_s2 = extra_smooth_q(q_s, joints=(4,5), win=11)

    # Re-derivar con preprocess_traj para tener dq, ddq y tau coherentes
    pad2 = 100
    q_ext = np.vstack([
        np.tile(q_s2[0, :], (pad2, 1)),
        q_s2,
        np.tile(q_s2[-1, :], (pad2, 1)),
    ])
    q_f, dq_f, ddq_f = ut.preprocess_traj(q_ext, DT)
    tau_f = limb.get_torq_traj(q_f, dq_f, ddq_f)

    return q_f[pad2:-pad2], dq_f[pad2:-pad2], ddq_f[pad2:-pad2], tau_f[pad2:-pad2]

def add_j4_sine_and_recompute(q, limb, amp_deg=10.0, cycles=2, pad=100):
    """
    Variante mixta J4–J5 sobre la trayectoria ya suavizada en J5/J6.
    """
    N = q.shape[0]
    t_norm = np.linspace(0.0, 1.0, N)
    delta_j4 = np.deg2rad(amp_deg) * np.sin(2.0 * np.pi * cycles * t_norm)

    q_mod = q.copy()
    q_mod[:, 3] += delta_j4

    q_ext = np.vstack([
        np.tile(q_mod[0, :], (pad, 1)),
        q_mod,
        np.tile(q_mod[-1, :], (pad, 1)),
    ])
    q_f, dq_f, ddq_f = ut.preprocess_traj(q_ext, DT)
    tau_f = limb.get_torq_traj(q_f, dq_f, ddq_f)

    return q_f[pad:-pad], dq_f[pad:-pad], ddq_f[pad:-pad], tau_f[pad:-pad]

def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))

def preview_energy(rows, tag="VAL"):
    M = np.vstack(rows)
    dq  = M[:, 7:14]
    ddq = M[:, 14:21]
    tau = M[:, 21:28]
    print(f"[PREVIEW:{tag}] J5: RMS(dq)={rms(dq[:,4]):.4f}  RMS(ddq)={rms(ddq[:,4]):.4f}  RMS(tau)={rms(tau[:,4]):.4f}")
    print(f"[PREVIEW:{tag}] J6: RMS(dq)={rms(dq[:,5]):.4f}  RMS(ddq)={rms(ddq[:,5]):.4f}  RMS(tau)={rms(tau[:,5]):.4f}")

# ================ Main ================
def main():
    if os.path.exists(SAVE_CSV):
        os.remove(SAVE_CSV)

    set_up_world(1.0/DT)
    center, meta_center = load_center()
    print(f"[CENTER] Usando center_cache.json -> {center}  (phi={meta_center.get('phi')}, theta={meta_center.get('theta')})")
    print("[INFO] VALIDACIÓN J5–J6: misma lógica que TRAIN pero con movimientos algo distintos.")

    rows = []

    # -------- Paquete J5-focus --------
    print(f"\n[PAQUETE VAL] J5-focus (n_seg={N_SEG_J5})")
    for T in T_LIST_J5:
        q, dq, ddq, tau = make_segments_fixed_pos(
            center, T,
            ORIENT_AMPS_J5, PHASES_J5,
            REPEAT_J5, N_SEG_J5,
            limb_left
        )
        rows.append(np.hstack([q, dq, ddq, tau]))
        print(f"  [J5-VAL]     T={T:.2f}s  N={len(q)}")

        q_mix, dq_mix, ddq_mix, tau_mix = add_j4_sine_and_recompute(q, limb_left)
        rows.append(np.hstack([q_mix, dq_mix, ddq_mix, tau_mix]))
        print(f"  [J4+J5-VAL] T={T:.2f}s  N={len(q_mix)}")

    # -------- Paquete J6-focus --------
    print(f"\n[PAQUETE VAL] J6-focus (n_seg={N_SEG_J6})")
    for T in T_LIST_J6:
        q, dq, ddq, tau = make_segments_fixed_pos(
            center, T,
            ORIENT_AMPS_J6, PHASES_J6,
            REPEAT_J6, N_SEG_J6,
            limb_left
        )
        rows.append(np.hstack([q, dq, ddq, tau]))
        print(f"  [J6-VAL]     T={T:.2f}s  N={len(q)}")

    preview_energy(rows, tag="VAL")
    stack_and_save(rows, SAVE_CSV)

    meta = dict(
        kind="wrist56_val_orient_only",
        dt=DT, phi=PHI, theta=THETA,
        center=list(map(float, center)),
        center_source=CENTER_CACHE,
        J5=dict(
            T_list=T_LIST_J5,
            orient_amps_deg=ORIENT_AMPS_J5,
            phases_deg=PHASES_J5,
            repeat=REPEAT_J5,
            n_seg=N_SEG_J5,
            mixed_j4=True,
        ),
        J6=dict(
            T_list=T_LIST_J6,
            orient_amps_deg=ORIENT_AMPS_J6,
            phases_deg=PHASES_J6,
            repeat=REPEAT_J6,
            n_seg=N_SEG_J6
        ),
        file=SAVE_CSV
    )
    with open(SAVE_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[OK] META: {SAVE_META}")

if __name__ == "__main__":
    main()

