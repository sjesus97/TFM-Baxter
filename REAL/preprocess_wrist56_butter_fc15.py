
import os
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

DT_DEFAULT = 1.0 / 500.0
NUM_JOINTS = 7
EPS = 1e-12

def load_matrix_no_header(path, num_joints=7):
    """
    Lee position.csv / effort.csv sin cabecera.
    Autodetecta separador (espacios, tabs, comas).
    Devuelve (N, num_joints).
    """
    df = pd.read_csv(path, header=None, sep=None, engine="python")
    arr = df.to_numpy(dtype=float)
    if arr.shape[1] < num_joints:
        raise ValueError(f"{path}: se esperaban >= {num_joints} columnas y hay {arr.shape[1]}")
    return arr[:, :num_joints]

def deriv_central_mat(X, dt):
    dX = np.zeros_like(X, dtype=float)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0]    = (X[1]  - X[0])  / dt
    dX[-1]   = (X[-1] - X[-2]) / dt
    return dX

def butter_lp_filtfilt(X, fs, fc, order=4):
    if fc <= 0 or fc >= fs / 2:
        raise ValueError(f"fc inválida: {fc}. Debe estar en (0, fs/2). fs={fs}")
    wn = fc / (fs / 2.0)
    b, a = butter(order, wn, btype="low")
    Y = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        Y[:, j] = filtfilt(b, a, X[:, j], method="pad")
    return Y

def preprocess_q_to_dq_ddq(q, dt, fc, order):
    """
    Pipeline definitivo:
      q -> LP -> dq = d/dt -> LP -> ddq = d/dt -> LP
    """
    fs = 1.0 / dt
    q_f  = butter_lp_filtfilt(q, fs=fs, fc=fc, order=order)
    dq   = deriv_central_mat(q_f, dt)
    dq_f = butter_lp_filtfilt(dq, fs=fs, fc=fc, order=order)
    ddq  = deriv_central_mat(dq_f, dt)
    ddq_f= butter_lp_filtfilt(ddq, fs=fs, fc=fc, order=order)
    return q_f, dq_f, ddq_f

def build_output_df(q_f, dq_f, ddq_f, tau, num_joints=7):
    data = {}
    for j in range(num_joints):
        data[f"q{j+1}"] = q_f[:, j]
    for j in range(num_joints):
        data[f"dq{j+1}"] = dq_f[:, j]
    for j in range(num_joints):
        data[f"ddq{j+1}"] = ddq_f[:, j]
    for j in range(num_joints):
        data[f"tau{j+1}"] = tau[:, j]
    return pd.DataFrame(data)

def process_folder(base_dir, folder_name, out_dir, dt, fc, order, filter_tau=False):
    folder_path = os.path.join(base_dir, folder_name)
    pos_path = os.path.join(folder_path, "position.csv")
    eff_path = os.path.join(folder_path, "effort.csv")

    if not os.path.isfile(pos_path):
        raise FileNotFoundError(pos_path)
    if not os.path.isfile(eff_path):
        raise FileNotFoundError(eff_path)

    print(f"\n[PROC] {folder_name}")
    print(f"  position: {pos_path}")
    print(f"  effort  : {eff_path}")

    q = load_matrix_no_header(pos_path, NUM_JOINTS)
    tau = load_matrix_no_header(eff_path, NUM_JOINTS)

    N = min(len(q), len(tau))
    if len(q) != len(tau):
        print(f"  [WARN] longitudes distintas q={len(q)} tau={len(tau)} -> recorto a N={N}")
    q = q[:N]
    tau = tau[:N]

    # Preprocesado q->dq->ddq (definitivo)
    q_f, dq_f, ddq_f = preprocess_q_to_dq_ddq(q, dt=dt, fc=fc, order=order)

    # Tau: crudo o filtrado sin fase (opcional)
    tau_used = tau
    tag = "tauRAW"
    if filter_tau:
        fs = 1.0 / dt
        tau_used = butter_lp_filtfilt(tau, fs=fs, fc=fc, order=order)
        tag = "tauFILT"

    out_df = build_output_df(q_f, dq_f, ddq_f, tau_used, num_joints=NUM_JOINTS)

    out_name = f"{folder_name}_butter_fc{int(fc)}_{tag}.csv"
    out_path = os.path.join(out_dir, out_name)
    out_df.to_csv(out_path, index=False)

    print(f"  [OK] Guardado: {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="Directorio que contiene las carpetas wrist56_*")
    ap.add_argument("--out_dir", required=True, help="Directorio salida CSV")
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--fc", type=float, default=15.0)
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--filter_tau", action="store_true", help="Filtrar tau con Butter+filtfilt (cero fase)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    folders = [
        "wrist56_train_basic_articular",
        "wrist56_val_like_train_articular",
    ]

    for f in folders:
        process_folder(
            base_dir=args.base_dir,
            folder_name=f,
            out_dir=args.out_dir,
            dt=args.dt,
            fc=args.fc,
            order=args.order,
            filter_tau=args.filter_tau
        )

if __name__ == "__main__":
    main()

