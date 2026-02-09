#!/usr/bin/env python3
"""
vis_q_mj_bdh_multi_tasks_comp_amass_inertialization_fast.py

Fixes:
1) Prevent accidental ff/gg outputs:
   - Require BOTH Motion IDs to have known labels from comparison CSV
   - Enforce suffix in {"fg","gf"}; otherwise raise (so no ff/gg files are created)
2) Make filename consistent with the caller (run_vis_pairs.sh / data_gen_hrl_npy.py):
   - If OUTPUT_NPY_NAME is provided via env, it is used verbatim.
3) Add robust sanity checks:
   - Motion index range check against amass_train.pkl key list length
   - Helpful debug prints when mapping/motion index fails
"""

import os
import re
import csv
import joblib
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

# Global state (Original 유지)
motion_id, time_step, dt = 0, 0, 1 / 30
motion_data_all = []
motion_lengths = []

class SpringDamper:
    def __init__(self, dt, halflife=0.1):
        self.dt = dt
        self.d = 4.0 * np.log(2.0) / halflife
        self.k = self.d * self.d / 4.0
        self.pos_offset = None
        self.vel_offset = None
        self.active = False

    def trigger(self, curr_pos, curr_vel, target_pos, target_vel):
        self.pos_offset = curr_pos - target_pos
        self.vel_offset = curr_vel - target_vel
        self.active = True

    def update(self):
        if not self.active:
            return (np.zeros_like(self.pos_offset) if self.pos_offset is not None else 0.0), \
                   (np.zeros_like(self.vel_offset) if self.vel_offset is not None else 0.0)
        acc = -self.k * self.pos_offset - self.d * self.vel_offset
        self.vel_offset += acc * self.dt
        self.pos_offset += self.vel_offset * self.dt
        if np.max(np.abs(self.pos_offset)) < 1e-4 and np.max(np.abs(self.vel_offset)) < 1e-3:
            self.active = False
        return self.pos_offset.copy(), self.vel_offset.copy()

def quat_to_rotvec(q):
    r = R.from_quat([q[1], q[2], q[3], q[0]])
    return r.as_rotvec()

def compute_velocity(curr, prev, dt):
    return (curr - prev) / dt

def _extract_suffix_from_run_vis_pairs_sh(motion_indices: list, script_path: str) -> str:
    """
    Parse run_vis_pairs.sh to find the expected filename suffix (fg or gf) for the given MOTION_INDICES.
    This ensures we use the same order as defined in the shell script, not CSV labels.
    
    The shell script format is:
        echo "[N/1104] Running saved_desired_states_{id1}_{id2}_{suffix}.npy (MOTION_INDICES={id1},{id2})"
        MOTION_INDICES="{id1},{id2}" python ...
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"run_vis_pairs.sh not found: {script_path}")
    
    # Create search pattern: MOTION_INDICES="id1,id2"
    ids_str = ",".join(str(i) for i in motion_indices)
    pattern = rf'MOTION_INDICES="{re.escape(ids_str)}"'
    
    # Read all lines and search for the pattern
    with open(script_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if pattern in line:
            # Check the previous line (echo statement) which contains the expected filename
            if i > 0:
                prev_line = lines[i-1]
                # Extract filename from: echo "[N/1104] Running saved_desired_states_{id1}_{id2}_{suffix}.npy ..."
                match = re.search(r'saved_desired_states_\d+_\d+_(fg|gf)\.npy', prev_line)
                if match:
                    return match.group(1)
            # Also check current line in case format is different
            match = re.search(r'saved_desired_states_\d+_\d+_(fg|gf)\.npy', line)
            if match:
                return match.group(1)
    
    raise ValueError(
        f"Could not find expected filename for MOTION_INDICES={motion_indices} in {script_path}. "
        f"Make sure the shell script contains this pair. Searched for pattern: {pattern}"
    )

def _load_id_to_label_from_comparison_csv(comparison_csv_path: str) -> dict:
    """
    Mirror the labeling logic used in data_gen_hrl_npy.py:
    - Only label a motion if the 'better' method succeeded AND reward>0.
    - f: Falcon better; g: GMT better
    """
    if not os.path.exists(comparison_csv_path):
        raise FileNotFoundError(f"Comparison CSV not found: {comparison_csv_path}")

    id_to_label = {}
    with open(comparison_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                csv_motion_id = int(row["Motion ID"])
                diff_str = row.get("Reward Difference (Falcon - GMT)", "").strip()
                if not diff_str:
                    continue
                diff = float(diff_str)

                better = row.get("Better Method", "").strip().lower()
                if not better:
                    better = "falcon" if diff > 0 else "gmt"

                if better == "falcon":
                    succ = row.get("Falcon Success", "").strip().lower() == "true"
                    r_str = row.get("Falcon Reward", "").strip()
                    if succ and r_str:
                        r = float(r_str)
                        if r > 0:
                            id_to_label[csv_motion_id] = "f"
                elif better == "gmt":
                    succ = row.get("GMT Success", "").strip().lower() == "true"
                    r_str = row.get("GMT Reward", "").strip()
                    if succ and r_str:
                        r = float(r_str)
                        if r > 0:
                            id_to_label[csv_motion_id] = "g"
            except Exception:
                continue
    return id_to_label

def main():
    global motion_id, time_step, dt, motion_data_all, motion_lengths

    # Reset global state
    motion_id = 0
    time_step = 0

    motion_file = "/home/baekdh/dh_workspace/data_phc/data/amass/valid_jh/amass_train.pkl"
    humanoid_xml = "/home/baekdh/dh_workspace/hrl/data_generation/assets/g1_kungfu.xml"
    inertial_halflife = 0.2

    # 1) Parse env motion indices
    env_motion_indices = os.environ.get("MOTION_INDICES", "83,201")
    motion_indices = [int(p) for p in re.split(r"[,\s]+", env_motion_indices.strip()) if p]

    if len(motion_indices) < 2:
        raise ValueError(f"MOTION_INDICES must contain >=2 ids, got: {motion_indices}")

    # 2) Load AMASS motions by integer index (sanity-check index ranges)
    motion_dict = joblib.load(motion_file)
    available_keys = list(motion_dict.keys())
    n_keys = len(available_keys)

    bad = [i for i in motion_indices if (i < 0 or i >= n_keys)]
    if bad:
        raise IndexError(
            f"[ERROR] Some MOTION_INDICES are out of range for '{motion_file}'.\n"
            f"  n_keys={n_keys}\n"
            f"  bad_indices={bad}\n"
            f"  all_indices={motion_indices}\n"
            f"  tip: comparison CSV Motion ID numbering must match THIS pickle's key ordering.\n"
        )

    motion_keys = [available_keys[i] for i in motion_indices]
    motion_data_all = [motion_dict[k] for k in motion_keys]
    motion_lengths = [m["dof"].shape[0] for m in motion_data_all]

    # 3) Determine filename suffix from run_vis_pairs.sh (not from CSV)
    # This ensures we use the exact same order as defined in the shell script
    env_output_name = os.environ.get("OUTPUT_NPY_NAME", "").strip()
    
    if len(motion_indices) == 2:
        if env_output_name:
            npy_filename = env_output_name
        else:
            # Extract suffix (fg or gf) from run_vis_pairs.sh based on MOTION_INDICES order
            script_path = os.path.join(os.path.dirname(__file__), "run_vis_pairs_test.sh")
            suffix = _extract_suffix_from_run_vis_pairs_sh(motion_indices, script_path)
            if suffix not in ("fg", "gf"):
                raise ValueError(
                    f"[ERROR] Invalid suffix '{suffix}' extracted from run_vis_pairs.sh for ids={motion_indices}. "
                    f"Expected 'fg' or 'gf' only."
                )
            npy_filename = f"saved_desired_states_{motion_indices[0]}_{motion_indices[1]}_{suffix}.npy"
    else:
        if env_output_name:
            npy_filename = env_output_name
        else:
            # For multi-id sequences, still need to use CSV (not in run_vis_pairs.sh)
            comparison_csv_path = os.path.join(os.path.dirname(__file__), "..", "logs_eval", "single", "falcon_vs_gmt_comparison.csv")
            id_to_label = _load_id_to_label_from_comparison_csv(comparison_csv_path)
            missing = [mid for mid in motion_indices if mid not in id_to_label]
            if missing:
                raise KeyError(
                    f"[ERROR] Motion IDs not found in comparison CSV label map: {missing}\n"
                    f"  CSV: {comparison_csv_path}\n"
                )
            lbls = [id_to_label[mid] for mid in motion_indices]
            if any(lbls[i] == lbls[i+1] for i in range(len(lbls)-1)):
                raise ValueError(f"[ERROR] Multi-id labels include same-adjacent labels: {lbls}. Refusing to create.")
            npy_filename = f"saved_desired_states_{'_'.join(str(i) for i in motion_indices)}_{''.join(lbls)}.npy"

    comparison_dir = os.path.join(os.path.dirname(__file__), "..", "humanoidverse", "data", "motions", "sequence_data", "comparison_inertialization_diff_motions")
    os.makedirs(comparison_dir, exist_ok=True)
    npy_path = os.path.join(comparison_dir, npy_filename)

    # 4) Setup MuJoCo (No rendering)
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)

    inertializer_pos = SpringDamper(dt, inertial_halflife)
    inertializer_rot = SpringDamper(dt, inertial_halflife)
    inertializer_dof = SpringDamper(dt, inertial_halflife)

    # 5) Sequential Alignment
    print("Aligning motions...")
    prev_q_end = motion_data_all[0]["root_rot"][-1]
    prev_rpy = R.from_quat([prev_q_end[1], prev_q_end[2], prev_q_end[3], prev_q_end[0]]).as_euler("xyz")
    global_end_pos = motion_data_all[0]["root_trans_offset"][-1].copy()

    for i in range(1, len(motion_data_all)):
        m = motion_data_all[i]
        curr_q_s = m["root_rot"][0]
        curr_rpy = R.from_quat([curr_q_s[1], curr_q_s[2], curr_q_s[3], curr_q_s[0]]).as_euler("xyz")
        delta_roll = prev_rpy[0] - curr_rpy[0]

        orig_first_p = m["root_trans_offset"][0].copy()
        for t in range(len(m["root_rot"])):
            rpy = R.from_quat([m["root_rot"][t][1], m["root_rot"][t][2], m["root_rot"][t][3], m["root_rot"][t][0]]).as_euler("xyz")
            rpy[0] += delta_roll
            nq = R.from_euler("xyz", rpy).as_quat()
            m["root_rot"][t] = [nq[3], nq[0], nq[1], nq[2]]
            m["root_trans_offset"][t] = orig_first_p + R.from_euler("z", delta_roll).apply(m["root_trans_offset"][t] - orig_first_p)

        m["root_trans_offset"] += (global_end_pos - m["root_trans_offset"][0])
        global_end_pos = m["root_trans_offset"][-1].copy()

        prev_rpy = R.from_quat([m["root_rot"][-1][1], m["root_rot"][-1][2], m["root_rot"][-1][3], m["root_rot"][-1][0]]).as_euler("xyz")

    # 6) Generation Loop
    saved_states = []
    global_time = 0.0

    print(f"Generating: {npy_filename}")
    print(f"  Motion IDs: {motion_indices}")
    print(f"  Motion keys: {motion_keys}")
    print(f"  Motion lengths: {motion_lengths}")
    print(f"  Save path: {npy_path}")

    while motion_id < len(motion_data_all):
        curr_m = motion_data_all[motion_id]
        curr_idx = int(time_step / dt)

        if curr_idx >= motion_lengths[motion_id]:
            if motion_id < len(motion_data_all) - 1:
                p_e = curr_m["root_trans_offset"][-1]
                p_v = compute_velocity(p_e, curr_m["root_trans_offset"][-2], dt)

                r_e = curr_m["root_rot"][-1]
                r_v = compute_velocity(quat_to_rotvec(r_e), quat_to_rotvec(curr_m["root_rot"][-2]), dt)

                d_e = curr_m["dof"][-1]
                d_v = compute_velocity(d_e, curr_m["dof"][-2], dt)

                next_m = motion_data_all[motion_id + 1]
                p_s = next_m["root_trans_offset"][0]
                p_sv = compute_velocity(next_m["root_trans_offset"][1], p_s, dt)

                r_s = next_m["root_rot"][0]
                r_sv = compute_velocity(quat_to_rotvec(next_m["root_rot"][1]), quat_to_rotvec(r_s), dt)

                d_s = next_m["dof"][0]
                d_sv = compute_velocity(next_m["dof"][1], d_s, dt)

                inertializer_pos.trigger(p_e, p_v, p_s, p_sv)
                inertializer_dof.trigger(d_e, d_v, d_s, d_sv)

                r_off = (R.from_quat([r_e[1], r_e[2], r_e[3], r_e[0]]) * R.from_quat([r_s[1], r_s[2], r_s[3], r_s[0]]).inv()).as_rotvec()
                inertializer_rot.trigger(r_off, r_v - r_sv, np.zeros(3), np.zeros(3))

                motion_id += 1
                time_step = 0
                curr_m = motion_data_all[motion_id]
                curr_idx = 0
            else:
                break

        off_p, _ = inertializer_pos.update()
        off_r_v, _ = inertializer_rot.update()
        off_d, _ = inertializer_dof.update()

        t_r_obj = R.from_quat([curr_m["root_rot"][curr_idx][1], curr_m["root_rot"][curr_idx][2], curr_m["root_rot"][curr_idx][3], curr_m["root_rot"][curr_idx][0]])
        f_q = R.from_rotvec(t_r_obj.as_rotvec() + off_r_v).as_quat()
        f_r = np.array([f_q[3], f_q[0], f_q[1], f_q[2]])

        mj_data.qpos[:3] = curr_m["root_trans_offset"][curr_idx] + off_p
        mj_data.qpos[3:7] = f_r[[3, 0, 1, 2]]
        mj_data.qpos[7:] = curr_m["dof"][curr_idx] + off_d

        state = np.concatenate([[global_time],
                                mj_data.qpos[:3].copy(),
                                mj_data.qpos[3:7].copy(),
                                mj_data.qpos[7:].copy(),
                                [motion_id]])
        saved_states.append(state)

        mujoco.mj_forward(mj_model, mj_data)
        time_step += dt
        global_time += dt

    if len(saved_states) == 0:
        raise RuntimeError(f"No states were generated; refusing to save empty file: {npy_filename}")

    saved_states = np.asarray(saved_states)
    if saved_states.ndim == 1:
        saved_states = saved_states.reshape(1, -1)

    np.save(npy_path, saved_states)
    print(f"Success! Saved {saved_states.shape[0]} frames to '{npy_path}' (shape: {saved_states.shape})")

if __name__ == "__main__":
    main()
