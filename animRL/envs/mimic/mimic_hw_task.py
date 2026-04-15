import sys
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *
import torch

from animRL import ROOT_DIR
from animRL.dataloader.motion_loader import MotionLoader
from animRL.envs.base.base_task import BaseTask
from animRL.utils.math import *
from animRL.utils.plots import forward_kinematics, plot_robot

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from urdfpy import URDF


class MimicHWTask(BaseTask):
    """Mimic a reference motion on the mini-pi hardware stack."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        self.motion_loader = MotionLoader(device=self.device,
                                          sim_frame_dt=self.dt,
                                          cfg=cfg.motion_loader,
                                          num_joints=cfg.env.num_actions,
                                          num_ee=len(cfg.asset.ee_offsets.keys())
                                          )
        assert self.motion_loader.num_motions == 1, "Only one motion clip is supported"
        self.data['motion_loader'] = self.motion_loader
        self.phase_rate = 1.0 / self.motion_loader.trajectory_num_frames[0]
        self.target_frames = self.motion_loader.get_frame_at_phase(self.phase)

        self.RSI = self.cfg.env.reference_state_initialization
        self.push_robots = self.cfg.domain_rand.push_robots
        if self.push_robots:
            self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        self.reset_triggered = False
        self._prepare_reward_function()
        self._setup_fake_camera_img()
        self._validate_config()

    def _validate_config(self):
        self.post_physics_step()
        assert self.obs_buf.size(
            dim=1) == self.num_obs, f"Obs size should be {self.obs_buf.size(dim=1)} but is {self.num_obs}"

    def _init_buffers(self):
        super()._init_buffers()

        self.play_step = 0
        self.reset_phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.single_obs_dim = self.cfg.env.num_observations
        self.obs_history_buf = torch.zeros(
            self.num_envs, self.obs_history_len, self.single_obs_dim, dtype=torch.float, device=self.device
        )
        self.data = {
            'num_ee': self.num_ee,
            'num_dof': self.num_dof,
            'env_origins': self.env_origins,
        }

        self.plot_robot = None

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        # fill extras
        self.extras["episode"] = {}
        eps = np.finfo(float).eps
        for key in self.episode_term_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_term_sums[key][env_ids] / (self.episode_length_buf[env_ids].float() + eps))
            self.episode_term_sums[key][env_ids] = 0.

        self.extras["episode"]['total_reward'] = torch.mean(
            self.episode_rew_sums[env_ids] / (self.episode_length_buf[env_ids].float() + eps))
        self.episode_rew_sums[env_ids] = 0.

        self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # reset robot states
        if self.RSI:
            self.reset_phase[env_ids] = torch_rand_float(0., 1., (len(env_ids), 1), device=self.device).squeeze(1)
            self.phase[env_ids] = self.reset_phase[env_ids]
            self.target_frames[env_ids] = self.motion_loader.get_frame_at_phase(self.reset_phase[env_ids])
            self._reset_dofs_from_frames(env_ids, self.target_frames[env_ids])
            self._reset_root_states_from_frames(env_ids, self.target_frames[env_ids])

        else:
            # ----------- TODO 1.2: Manage phase reset

            # ----------- End of implementation

            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

        self._refresh_quantities()

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.obs_history_buf[env_ids] = 0.

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self.post_physics_step()

    def _refresh_quantities(self):
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.data['root_states'] = self.root_states
        self.data['base_quat'] = self.base_quat
        self.data['base_lin_vel'] = self.base_lin_vel
        self.data['base_ang_vel'] = self.base_ang_vel
        self.data['projected_gravity'] = self.projected_gravity
        self.data['dof_pos'] = self.dof_pos
        self.data['dof_vel'] = self.dof_vel

        # end effector
        ee_global_ = self.rigid_body_state[:, self.ee_indices, 0:3]
        ee_quat = self.rigid_body_state[:, self.ee_indices, 3:7]
        ee_global = torch.zeros_like(ee_global_)
        ee_local = torch.zeros_like(ee_global_)
        for i in range(len(self.ee_indices)):
            ee_global[:, i, :] = ee_global_[:, i, :] + quat_rotate(ee_quat[:, i, :],
                                                                   self.ee_offsets[i, :].unsqueeze(0).repeat(
                                                                       self.num_envs, 1))
            ee_local_ = ee_global[:, i, :].squeeze() - self.root_states[:, 0:3]
            ee_local[:, i, :] = quat_rotate_inverse(self.base_quat, ee_local_)

        self.data['ee_global'] = ee_global
        self.data['ee_local'] = ee_local

        self.data['phase'] = self.phase
        self.data['reset_phase'] = self.reset_phase
        self.data['target_frames'] = self.target_frames
        self.data['reset_frames'] = self.motion_loader.get_frame_at_phase(self.reset_phase)
        self.data['joint_targets_rate'] = torch.norm(self.last_joint_targets - self.joint_targets, p=2,
                                                     dim=1) / self.joint_targets_rate_scaler

    def pre_physics_step(self, actions):
        self.actions = actions
        clip_joint_target = self.cfg.control.clip_joint_target
        scale_joint_target = self.cfg.control.scale_joint_target
        self.joint_targets = torch.clip(actions * scale_joint_target, -clip_joint_target, clip_joint_target).to(
            self.device)

    def step(self, actions):
        self.pre_physics_step(actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            if self.cfg.asset.default_dof_drive_mode == 3:  # torque mode
                self.torques = self._compute_torques(self.joint_targets).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))

            if self.cfg.asset.default_dof_drive_mode == 1:  # position mode
                pos_targets = self.joint_targets + self.default_dof_pos
                self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(pos_targets))

            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()
        return self.obs_buf, self.critic_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        # contact termination
        self.reset_buf = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 0.1,
            dim=1)

        # reach maximal velocity termination
        self.reset_buf |= torch.norm(self.base_lin_vel, dim=-1) > self.cfg.termination.max_base_lin_vel
        self.reset_buf |= torch.norm(self.base_ang_vel, dim=-1) > self.cfg.termination.max_base_ang_vel
        self.reset_buf |= torch.abs(self.root_states[:, 2]) > self.cfg.termination.max_height

        # time out
        self.time_out_buf = self.episode_length_buf > self.max_episode_length

        # ----------- TODO 1.2: Implement additional timeout conditions

        # ----------- End of implementation

        self.reset_buf |= self.time_out_buf

        if self.reset_triggered:
            self.reset_buf[:] = 1
            self.reset_triggered = False

    def post_physics_step(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.episode_length_buf += 1
        self.play_step += 1

        self._refresh_quantities()

        self.check_termination()
        self.compute_reward()

        # ----------- TODO 1.2: manage phase for next step

       # ----------- End of implementation

        # self.process_keystroke()

        if self.push_robots and (self.play_step % self.push_interval == 0):
            self._push_robots()

        env_ids_reset = self.reset_buf.nonzero().flatten()
        self.reset_idx(env_ids_reset)

        self.compute_observations()
        self.last_joint_targets[:] = self.joint_targets[:]
        self.last_actions[:] = self.actions[:]

    def compute_observations(self):
        current_obs = torch.zeros(self.num_envs, self.num_obs)

        # ----------- TODO 2.1: update observation

        # ----------- End of implementation

        self.obs_history_buf = torch.roll(self.obs_history_buf, shifts=-1, dims=1)
        self.obs_history_buf[:, -1, :] = current_obs
        self.obs_buf = self.obs_history_buf.reshape(self.num_envs, -1)

    def get_observations(self):
        return self.obs_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.8, 1.2,
                                                                        (len(env_ids), self.num_dof),
                                                                        device=self.device)
        self.dof_vel[env_ids] = 0.
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_dofs_from_frames(self, env_ids, frames):
        frames = frames.clone()
        self.dof_pos[env_ids] = self.motion_loader.get_joint_pose(frames)
        self.dof_vel[env_ids] = self.motion_loader.get_joint_vel(frames)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]

        # base velocities
        self.root_states[env_ids, 7:13] = 0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_from_frames(self, env_ids, frames):
        frames = frames.clone()
        root_pos = self.motion_loader.get_root_pos(frames)
        root_pos[:, :2] = self.env_origins[env_ids, :2]
        root_pos[:, 2] += self.cfg.init_state.added_height  # add height

        self.root_states[env_ids, :3] = root_pos
        root_orn = self.motion_loader.get_root_rot(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, self.motion_loader.get_linear_vel(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, self.motion_loader.get_angular_vel(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states_for_replay(self, env_ids, frames):
        frames = frames.clone()
        root_pos = self.motion_loader.get_root_pos(frames)
        root_pos[:, :2] += self.env_origins[env_ids, :2]

        root_pos[:, 2] += self.cfg.init_state.added_height  # add height
        self.root_states[env_ids, :3] = root_pos
        root_orn = self.motion_loader.get_root_rot(frames)
        self.root_states[env_ids, 3:7] = root_orn
        self.root_states[env_ids, 7:10] = quat_rotate(root_orn, self.motion_loader.get_linear_vel(frames))
        self.root_states[env_ids, 10:13] = quat_rotate(root_orn, self.motion_loader.get_angular_vel(frames))

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_envs_for_replay(self, env_ids, frames):
        self._reset_dofs_from_frames(env_ids, frames)
        self._reset_root_states_for_replay(env_ids, frames)
        self._draw_debug_vis(vis_flag=self.cfg.viewer.vis_flag, frame=frames)

    def _push_robots(self):
        max_vel = self.cfg.domain_rand.max_push_vel_xyz
        max_avel = self.cfg.domain_rand.max_push_avel_xyz

        # ----------- TODO 3.1: implement push robots

        # ----------- End of implementation

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def get_time_stamp(self):
        return self.play_step * self.dt

    def replay_motion(self):
        self.play_step = 0
        traj_idxs = np.arange(self.num_envs)

        while True:
            # self.process_keystroke()
            if self.is_playing:
                frame_id = np.minimum(self.play_step, self.motion_loader.trajectory_num_frames - 1).astype(int)
                reference_frames = torch.vstack(
                    [self.motion_loader.trajectories_full[traj_id][frame_id[traj_id]] for traj_id in traj_idxs])

                self.reset_envs_for_replay(torch.arange(self.num_envs, device=self.device), reference_frames)
                self.play_step += 1
            if self.reset_triggered:
                self.play_step = 0
                self.reset_triggered = False
            self.render(replay=True)

    def _draw_debug_vis(self, vis_flag='', frame=None):
        """ Draws visualizations for debugging (slows down simulation a lot).
            vis_flag options: 'ground_truth', 'end_effector'.
            include 'ref_only' in vis_flag if you want to draw only for ref env
        """
        self.gym.clear_lines(self.viewer)
        focus_env = self.cfg.viewer.ref_env
        color_red = (1, 0, 0)
        color_green = (0, 1, 0)
        color_blue = (0, 0, 1)
        color_white = (0, 0, 0)
        color_black = (0.2, 0.2, 0.2)
        color_yellow = (1, 1, 0)

        if 'end_effector' in vis_flag:
            if frame is None:
                # draw global ee pos
                sphere_geom_green = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_green)
                sphere_geom_red = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_red)
                for env_id in range(self.num_envs):
                    if env_id == self.cfg.viewer.ref_env or 'ref_only' not in vis_flag:
                        ee_pos = self.data['ee_global'][env_id, :]
                        for j in range(self.num_ee):
                            x = ee_pos[j, 0]
                            y = ee_pos[j, 1]
                            z = ee_pos[j, 2]
                            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                            gymutil.draw_lines(sphere_geom_green, self.gym, self.viewer, self.envs[env_id], sphere_pose)
            else:
                # draw ee pos from reference frame
                sphere_geom = gymutil.WireframeSphereGeometry(0.03, 4, 4, None, color=color_blue)
                ee_pos_global_all = self.motion_loader.get_ee_pos_global(frame).reshape(
                    (-1, self.num_ee, 3)) + self.env_origins.unsqueeze(1).repeat(1, self.num_ee, 1)
                for env_id in range(self.num_envs):
                    if env_id == self.cfg.viewer.ref_env or 'ref_only' not in vis_flag:
                        ee_pos = ee_pos_global_all[env_id, :]
                        for j in range(self.num_ee):
                            x = ee_pos[j, 0]
                            y = ee_pos[j, 1]
                            z = ee_pos[j, 2]
                            sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[env_id], sphere_pose)

        if 'ground_truth' in vis_flag:
            reference_frames = self.target_frames[focus_env]
            reference_frames[2] += self.env_origins[focus_env, 2]
            pos = reference_frames[:2] - self.motion_loader.get_frame_at_phase(self.reset_phase)[self.focus_env, :2]
            pos += self.env_origins[focus_env, :2]
            pos[0] -= 1.0  # shift x
            pos[1] += 0.0  # shift y
            for i in range(self.num_bodies):
                self.gym.set_rigid_body_color(self.envs[-1], 0, i, gymapi.MESH_VISUAL, gymapi.Vec3(*color_green))
            self.env_origins[self.num_envs - 1] = self.env_origins[focus_env]
            self.draw_ghost(self.num_envs - 1, torch.cat([pos, reference_frames[2:7]]),
                            reference_frames[7:7 + self.num_dof])

    def render(self, sync_frame_time=True, replay=False):
        if not replay:
            self._draw_debug_vis(vis_flag=self.cfg.viewer.vis_flag)
        super().render(sync_frame_time=sync_frame_time)

    def _setup_fake_camera_img(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.plot_robot = URDF.load(self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR))
        self.canvas = FigureCanvas(self.fig)

    def fake_camera_img(self):
        self.ax.clear()
        joint_angles = self.dof_pos[self.focus_env].detach().cpu().numpy()
        joint_angles = {'names': self.dof_names,
                        'values': joint_angles}
        frames = forward_kinematics(self.plot_robot, self.root_states[self.focus_env, :3].detach().cpu().numpy(),
                                    self.root_states[self.focus_env, 3:7].detach().cpu().numpy(),
                                    joint_angles
                                    )
        plot_robot(self.ax, self.plot_robot, frames)
        self.canvas.draw()

        # Convert to numpy array
        img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(self.canvas.get_width_height()[::-1] + (3,))  # (H, W, 3)
        return img

    def process_keystroke(self):
        # check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value > 0:
                if evt.action == "QUIT":
                    sys.exit()
                if evt.action == "toggle_viewer_sync":
                    self.enable_viewer_sync = not self.enable_viewer_sync
                if evt.action == "toggle_overview":
                    self.overview = not self.overview

                if evt.action == "key_space":
                    self.is_playing = not self.is_playing
                if evt.action == "key_r":
                    self.reset_triggered = True

                if evt.action == "key_plus":
                    self.cfg.viewer.ref_env = np.clip(self.cfg.viewer.ref_env + 1, 0, self.num_envs - 1)
                    print("focus env: ", self.cfg.viewer.ref_env)
                if evt.action == "key_minus":
                    self.cfg.viewer.ref_env = np.clip(self.cfg.viewer.ref_env - 1, 0, self.num_envs - 1)
                    print("focus env: ", self.cfg.viewer.ref_env)
