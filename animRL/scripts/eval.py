# created by Fatemeh Zargarbashi - 2025
import isaacgym
from matplotlib.animation import FFMpegWriter

from animRL import ROOT_DIR
from animRL.utils.helpers import get_load_path, update_cfgs_from_dict, export_policy_as_onnx
from animRL.utils.isaac_helpers import get_args
from animRL.envs import task_registry
from animRL.utils.plots import forward_kinematics, animate_robot

import json
import os
import numpy as np
import torch
from urdfpy import URDF

"""
This code is used to evaluate the trained policy on the character. Run it with your corresponding task name and load_run. 
It will save a plot of the rewards, a video of the motion, and a json file containing the observations, rewards, and dones.
"""


class Eval:
    """ Evaluation script to play the policy."""

    def __init__(self, args, seed=1):
        self.args = args

        env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

        if args.load_run is not None:
            train_cfg.runner.load_run = args.load_run
        if args.checkpoint is not None:
            train_cfg.runner.checkpoint = args.checkpoint
        load_path = get_load_path(
            os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name),
            load_run=train_cfg.runner.load_run,
            checkpoint=train_cfg.runner.checkpoint,
        )
        self.load_path = load_path
        print(f"Loading model from: {load_path}")

        # load config
        load_config_path = os.path.join(os.path.dirname(load_path), f"{train_cfg.runner.experiment_name}.json")
        with open(load_config_path) as f:
            load_config = json.load(f)
            update_cfgs_from_dict(env_cfg, train_cfg, load_config)

        # overwrite config params
        env_cfg.seed = seed
        env_cfg.env.num_envs = 2

        env_cfg.env.play = True
        env_cfg.sim.use_gpu_pipeline = False
        env_cfg.env.episode_length = 500  # to prevent timeout
        env_cfg.viewer.overview = False
        env_cfg.viewer.vis_flag = ['ground_truth']

        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

        # prepare environment, runner and policy
        env, _ = task_registry.make_env(name=self.args.task, args=self.args, env_cfg=self.env_cfg)
        self.env = env
        self.runner = task_registry.make_alg_runner(env=env,
                                                    name=self.args.task,
                                                    args=self.args,
                                                    env_cfg=self.env_cfg,
                                                    train_cfg=self.train_cfg)
        self.runner.load(self.load_path)  # load policy
        self.policy = self.runner.get_inference_policy(device=self.env.device)
        # export_dir = os.path.join(os.path.dirname(self.load_path), "exported")
        # export_policy_as_onnx(self.runner.policy, export_dir,
        #                       self.runner.actor_obs_normalizer, filename=f"model.onnx")

        self.env.reset()
        self.obs = self.env.get_observations()

        self.info = None
        self.buf = {'obs': [],
                    'dones': [],
                    'rewards': {}
                    }
        # temporary buffers to collect tensors during rollout
        self._buf_tensors = {'obs': [], 'dones': [], 'rewards': {}}
        self.all_joint_angles = []
        self.all_base_pos = []
        self.all_base_quat = []
        self.all_ref_joint_angles = []
        self.all_ref_base_pos = []
        self.all_ref_base_quat = []

    def _add_data_to_buf(self, env_id):
        self.buf['obs'].append(self.obs[env_id])
        self.buf['dones'].append(self.dones[env_id])
        for key in self.env.rew_terms_buf:
            if key not in self.buf['rewards']:
                self.buf['rewards'][key] = []
            self.buf['rewards'][key].append(self.env.rew_terms_buf[key][env_id])

        self.all_base_pos.append(self.env.root_states[env_id, :3].clone())
        self.all_base_quat.append(self.env.root_states[env_id, 3:7].clone())
        self.all_joint_angles.append(self.env.dof_pos[env_id].clone())

        target_frame = self.env.data['target_frames'][env_id]
        self.all_ref_base_pos.append(self.env.motion_loader.get_root_pos(target_frame))
        self.all_ref_base_quat.append(self.env.motion_loader.get_root_rot(target_frame))
        self.all_ref_joint_angles.append(self.env.motion_loader.get_joint_pose(target_frame))

    def play(self, max_steps=500):
        self.env.is_playing = True
        x_getter = self.env.get_time_stamp
        y_getters = (self.env.getplt_rewards,)
        self.env.plotter_init(y_getters)

        # rollout
        env_id = 0
        with torch.no_grad():
            for i in range(max_steps):
                actions = self.policy(self.obs)
                self.obs, _, _, self.dones, self.info = self.env.step(actions)

                self._add_data_to_buf(env_id)
                self.env.plotter_update(i, x_getter, y_getters)

        json_path = os.path.join(os.path.dirname(self.load_path), "eval_buf.json")
        self.save_eval_buf(json_path)
        fig_path = os.path.join(os.path.dirname(self.load_path), "eval_rewards.png")
        self.save_plot(fig_path, y_getters)
        anim_path = os.path.join(os.path.dirname(self.load_path), "animation.mp4")
        self.save_animation(anim_path)

    def save_eval_buf(self, save_path):
        for key in self.buf:
            if key == 'rewards':
                for rew_key in self.buf['rewards']:
                    self.buf[key][rew_key] = torch.stack(self.buf[key][rew_key], dim=0).cpu().numpy().tolist()
            else:
                self.buf[key] = torch.stack(self.buf[key], dim=0).cpu().numpy().tolist()

        json.dump(self.buf, indent=2, fp=open(save_path, 'w'))
        print(f"Log data saved to {save_path}")

    def save_animation(self, anim_path=None):
        # animate robot
        base_pos = torch.stack(self.all_base_pos, dim=0).cpu().numpy()
        base_quat = torch.stack(self.all_base_quat, dim=0).cpu().numpy()
        joint_angles = [{'names': self.env.dof_names, 'values': t.cpu().numpy()} for t in self.all_joint_angles]
        ref_base_pos = torch.stack(self.all_ref_base_pos, dim=0).cpu().numpy()
        ref_base_quat = torch.stack(self.all_ref_base_quat, dim=0).cpu().numpy()
        ref_joint_angles = [{'names': self.env.dof_names, 'values': t.cpu().numpy()} for t in self.all_ref_joint_angles]

        urdf_path = self.env_cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        robot = URDF.load(urdf_path)

        n_frames = len(joint_angles)
        frames_list = [forward_kinematics(robot, base_pos[i], base_quat[i], joint_angles[i]) for i in range(n_frames)]
        ref_frames_list = [forward_kinematics(robot, ref_base_pos[i], ref_base_quat[i], ref_joint_angles[i]) for i in
                           range(n_frames)]

        ani = animate_robot(robot, frames_list, ref_frames_list)
        FFWriter = FFMpegWriter(fps=50)
        ani.save(anim_path, writer=FFWriter)
        print(f"Animation saved to {anim_path}")

    def save_plot(self, save_path, y_getters=None):
        for subplot_id in range(len(y_getters)):
            curr_plotter_ax = self.env.plotter_axs[subplot_id] if len(y_getters) > 1 else self.env.plotter_axs
            for line in self.env.plotter_lines[subplot_id]:
                curr_plotter_ax.add_line(line)
        self.env.plotter_fig.savefig(save_path)
        print(f"Plots saved to {save_path}")


if __name__ == '__main__':
    args = get_args()
    args.dv = True  # set to True to open simulator viewer (only works if you have a display)
    seed = 2  # Note: you can change the seed to get a different behavior
    ip = Eval(args, seed)
    ip.play()
    ip.runner.close()
