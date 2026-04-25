#!/usr/bin/env python3
"""Multi-robot sim2sim for PI variants in one MuJoCo scene (batched policy inference)."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from animRL import ROOT_DIR
from animRL.cfg.mimic.walk_hw_config import WalkHWCfg
from animRL.cfg.mimic.walk_hw_config import WalkHWTrainCfg
from animRL.utils.helpers import get_load_path, update_cfgs_from_dict
from animRL.runners.modules.normalizer import EmpiricalNormalization
from animRL.runners.modules.policy import Policy


def quat_rotate_inverse_xyzw(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_xyz = q_xyzw[:3]
    q_w = q_xyzw[3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_xyz, v) * (2.0 * q_w)
    c = q_xyz * (2.0 * np.dot(q_xyz, v))
    return a - b + c


def infer_hidden_dims_from_policy_dict(policy_state: dict) -> list[int]:
    layer_items = []
    for k, v in policy_state.items():
        if not k.startswith("policy_latent_net.") or not k.endswith(".weight"):
            continue
        idx_s = k.split(".")[1]
        if not idx_s.isdigit():
            continue
        layer_items.append((int(idx_s), tuple(v.shape)))
    if not layer_items:
        raise ValueError("Could not infer hidden dims from checkpoint policy_latent_net weights.")
    layer_items.sort(key=lambda x: x[0])
    return [shape[0] for _, shape in layer_items]


def load_pt_policy(
    pt_path: Path,
    num_obs: int,
    num_actions: int,
    activation: str,
) -> tuple[Policy, EmpiricalNormalization]:
    ckpt = torch.load(str(pt_path), map_location="cpu")
    if "policy_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint (missing 'policy_dict'): {pt_path}")

    policy_dict = ckpt["policy_dict"]
    hidden_dims = infer_hidden_dims_from_policy_dict(policy_dict)
    policy = Policy(
        num_obs=num_obs,
        num_actions=num_actions,
        hidden_dims=hidden_dims,
        activation=activation,
        log_std_init=0.0,
        device="cpu",
    )
    policy.load_state_dict(policy_dict, strict=True)
    policy.eval()

    normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).cpu()
    if "actor_obs_normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["actor_obs_normalizer"], strict=False)
    normalizer.eval()
    return policy, normalizer


def load_num_frames_from_motion(motion_file: Path) -> int:
    with motion_file.open("r", encoding="utf-8") as f:
        motion_json = json.load(f)
    frames = motion_json.get("Frames", [])
    if not isinstance(frames, list) or len(frames) == 0:
        raise ValueError(f"Motion file has no Frames: {motion_file}")
    return len(frames)


def get_training_pd_gains(cfg: WalkHWCfg, joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stiffness = cfg.control.stiffness
    damping = cfg.control.damping
    kp = np.zeros(len(joint_names), dtype=np.float64)
    kd = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        for key in stiffness.keys():
            if key in name:
                kp[i] = float(stiffness[key])
                kd[i] = float(damping[key])
                break
    return kp, kd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-robot MuJoCo sim2sim for PI variants.")
    parser.add_argument("--load_run", type=str, default=-1, help="Run folder to load. Use -1 for the latest run.")
    parser.add_argument("--checkpoint", type=int, default=-1, help="Checkpoint number to load. Use -1 for latest.")
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("animRL/resources/robots/pi_12dof/pi_12dof_multi_scene.xml"),
        help="Multi-robot MuJoCo scene XML.",
    )
    parser.add_argument("--num_robots", type=int, default=6, help="Number of robots in the scene.")
    parser.add_argument("--sim_duration", type=float, default=120.0, help="Simulation duration in seconds.")
    parser.add_argument("--sim_dt", type=float, default=0.005, help="MuJoCo physics dt.")
    parser.add_argument("--control_decimation", type=int, default=4, help="Sim steps per policy step.")
    parser.add_argument("--tau_limit", type=float, default=20.0, help="Torque clip (Nm).")
    parser.add_argument("--render_decimation", type=int, default=1, help="Render every N physics steps.")
    parser.add_argument("--no_viewer", action="store_true", help="Headless run.")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    env_cfg = WalkHWCfg()
    train_cfg = WalkHWTrainCfg()

    load_path = get_load_path(
        os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
        load_run=args.load_run,
        checkpoint=args.checkpoint,
    )
    load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
    with open(load_config_path, "r", encoding="utf-8") as f:
        load_config = json.load(f)
    update_cfgs_from_dict(env_cfg, train_cfg, load_config)
    print(f"[INFO] Loading policy from: {load_path}")

    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve()))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    model.opt.timestep = args.sim_dt

    num_robots = args.num_robots
    render_decimation = max(1, args.render_decimation)
    control_decimation = args.control_decimation

    base_joint_names = list(env_cfg.init_state.default_joint_angles.keys())
    num_actions = len(base_joint_names)
    kp_base, kd_base = get_training_pd_gains(env_cfg, base_joint_names)
    default_q_base = np.array([env_cfg.init_state.default_joint_angles[n] for n in base_joint_names], dtype=np.float64)

    joint_qpos_adr = np.zeros((num_robots, num_actions), dtype=np.int32)
    joint_qvel_adr = np.zeros((num_robots, num_actions), dtype=np.int32)
    free_qpos_adr = np.zeros(num_robots, dtype=np.int32)
    free_qvel_adr = np.zeros(num_robots, dtype=np.int32)
    root_xy = np.zeros((num_robots, 2), dtype=np.float64)

    # Resolve prefixed names in generated multi scene: r{idx}_<original_name>
    for r in range(num_robots):
        free_name = f"r{r}_floating_base_joint"
        jid_free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, free_name)
        if jid_free == -1:
            raise ValueError(f"Missing free joint in scene: {free_name}")
        free_qpos_adr[r] = int(model.jnt_qposadr[jid_free])
        free_qvel_adr[r] = int(model.jnt_dofadr[jid_free])
        root_xy[r] = data.qpos[free_qpos_adr[r] : free_qpos_adr[r] + 2]

        for j, jname in enumerate(base_joint_names):
            full = f"r{r}_{jname}"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, full)
            if jid == -1:
                raise ValueError(f"Missing joint in scene: {full}")
            joint_qpos_adr[r, j] = int(model.jnt_qposadr[jid])
            joint_qvel_adr[r, j] = int(model.jnt_dofadr[jid])

    # Batch policy setup
    single_obs_dim = int(env_cfg.env.num_observations)
    obs_history_len = int(env_cfg.env.obs_history_len)
    policy_obs_dim = single_obs_dim * obs_history_len
    obs_hist = np.zeros((num_robots, obs_history_len, single_obs_dim), dtype=np.float32)
    last_action = np.zeros((num_robots, num_actions), dtype=np.float32)
    current_action = np.zeros((num_robots, num_actions), dtype=np.float64)
    target_q = np.tile(default_q_base, (num_robots, 1))
    phase = np.zeros(num_robots, dtype=np.float64)

    motion_file = Path(env_cfg.motion_loader.motion_files.format(ROOT_DIR=ROOT_DIR)).resolve()
    pt_policy, pt_normalizer = load_pt_policy(
        Path(load_path),
        num_obs=policy_obs_dim,
        num_actions=num_actions,
        activation=train_cfg.policy.activation,
    )
    num_frames = load_num_frames_from_motion(motion_file)
    phase_rate = 1.0 / float(num_frames)

    def reset_all() -> None:
        mujoco.mj_resetData(model, data)
        for r in range(num_robots):
            reset_robot(r)
        np.copyto(data.qfrc_applied, 0.0)
        obs_hist[:] = 0.0
        last_action[:] = 0.0
        current_action[:] = 0.0
        target_q[:] = np.tile(default_q_base, (num_robots, 1))
        phase[:] = 0.0
        mujoco.mj_forward(model, data)

    def reset_robot(r: int) -> None:
        qa = free_qpos_adr[r]
        va = free_qvel_adr[r]
        data.qpos[qa : qa + 2] = root_xy[r]
        data.qpos[qa + 2] = float(env_cfg.init_state.pos[2])
        init_q_xyzw = np.array(env_cfg.init_state.rot, dtype=np.float64)
        data.qpos[qa + 3 : qa + 7] = np.array(
            [init_q_xyzw[3], init_q_xyzw[0], init_q_xyzw[1], init_q_xyzw[2]],
            dtype=np.float64,
        )
        data.qvel[va : va + 6] = 0.0
        for j in range(num_actions):
            data.qpos[joint_qpos_adr[r, j]] = default_q_base[j]
            data.qvel[joint_qvel_adr[r, j]] = 0.0

        obs_hist[r] = 0.0
        last_action[r] = 0.0
        current_action[r] = 0.0
        target_q[r] = default_q_base
        phase[r] = 0.0

    def compute_obs_batch() -> np.ndarray:
        for r in range(num_robots):
            qa = free_qpos_adr[r]
            va = free_qvel_adr[r]
            quat_wxyz = data.qpos[qa + 3 : qa + 7].copy()
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
            ang_vel_world = data.qvel[va + 3 : va + 6].astype(np.float64)
            base_ang_vel_local = quat_rotate_inverse_xyzw(quat_xyzw, ang_vel_world)
            projected_gravity = quat_rotate_inverse_xyzw(quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float64))

            q = np.array([data.qpos[joint_qpos_adr[r, j]] for j in range(num_actions)], dtype=np.float64)
            qd = np.array([data.qvel[joint_qvel_adr[r, j]] for j in range(num_actions)], dtype=np.float64)

            single_obs = np.concatenate(
                [
                    projected_gravity,
                    base_ang_vel_local,
                    (q - default_q_base),
                    qd,
                    last_action[r].astype(np.float64),
                    np.array([phase[r]], dtype=np.float64),
                ],
                axis=0,
            ).astype(np.float32)
            obs_hist[r, :-1] = obs_hist[r, 1:]
            obs_hist[r, -1] = single_obs
        return obs_hist.reshape(num_robots, -1)

    def apply_pd_batch() -> None:
        np.copyto(data.qfrc_applied, 0.0)
        for r in range(num_robots):
            q = np.array([data.qpos[joint_qpos_adr[r, j]] for j in range(num_actions)], dtype=np.float64)
            qd = np.array([data.qvel[joint_qvel_adr[r, j]] for j in range(num_actions)], dtype=np.float64)
            tau = kp_base * (target_q[r] - q) - kd_base * qd
            tau = np.clip(tau, -args.tau_limit, args.tau_limit)
            for j in range(num_actions):
                data.qfrc_applied[joint_qvel_adr[r, j]] = tau[j]

    def policy_infer_batch(obs_batch: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs_batch).float()
        with torch.no_grad():
            obs_n = pt_normalizer(obs_t)
            act_t = pt_policy.act_inference(obs_n)
        return act_t.cpu().numpy().astype(np.float64)

    reset_all()
    sim_steps = int(args.sim_duration / model.opt.timestep)

    if args.no_viewer:
        start = time.time()
        for step in range(sim_steps):
            if step % control_decimation == 0:
                if np.any(done):
                    for r in np.where(done)[0]:
                        reset_robot(int(r))
                    mujoco.mj_forward(model, data)
                obs_batch = compute_obs_batch()
                current_action[:] = policy_infer_batch(obs_batch)
                last_action[:] = current_action.astype(np.float32)
                target_offset = np.clip(
                    current_action * float(env_cfg.control.scale_joint_target),
                    -float(env_cfg.control.clip_joint_target),
                    float(env_cfg.control.clip_joint_target),
                )
                target_q[:] = default_q_base[None, :] + target_offset
                phase += phase_rate
                done = phase >= (1.0 - phase_rate)
            apply_pd_batch()
            mujoco.mj_step(model, data)
        print(f"[INFO] Headless multi rollout finished in {time.time() - start:.2f}s")
        return

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        center_x = float(np.mean(root_xy[:, 0]))
        center_y = float(np.mean(root_xy[:, 1]))
        viewer.cam.lookat[:] = [center_x, center_y, 0.5]
        viewer.cam.distance = 6
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -30.0
        start = time.time()
        step = 0
        while viewer.is_running() and step < sim_steps:
            step_start = time.time()
            if step % control_decimation == 0:
                obs_batch = compute_obs_batch()
                current_action[:] = policy_infer_batch(obs_batch)
                last_action[:] = current_action.astype(np.float32)
                target_offset = np.clip(
                    current_action * float(env_cfg.control.scale_joint_target),
                    -float(env_cfg.control.clip_joint_target),
                    float(env_cfg.control.clip_joint_target),
                )
                target_q[:] = default_q_base[None, :] + target_offset
                phase += phase_rate
                done = phase >= (1.0 - phase_rate)
                if np.any(done):
                    for r in np.where(done)[0]:
                        reset_robot(int(r))
                    mujoco.mj_forward(model, data)
            apply_pd_batch()
            mujoco.mj_step(model, data)
            if step % render_decimation == 0:
                viewer.sync()
            step += 1
            dt_left = model.opt.timestep - (time.time() - step_start)
            if dt_left > 0:
                time.sleep(dt_left)

    print(f"[INFO] Multi rollout finished in {time.time() - start:.2f}s")


if __name__ == "__main__":
    run(parse_args())
