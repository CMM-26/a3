# created by Fatemeh Zargarbashi - 2026

from animRL.cfg.base.base_config import BaseEnvCfg, BaseTrainCfg


class MimicCfg(BaseEnvCfg):
    class env(BaseEnvCfg.env):
        num_observations = 1  # should be overwritten
        num_actions = 1  # should be overwritten
        num_privileged_obs = None  # None
        obs_history_len = 1

        episode_length = 250  # episode length

        reference_state_initialization = False  # initialize state from reference data

        play = False
        debug = False

    class motion_loader:
        motion_files = ''

    class init_state(BaseEnvCfg.init_state):
        pos = [0.0, 0.0, 0.345]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        added_height = 0.0  # height added to root when rsi

        # default joint angles =  target angles [rad] when action = 0.0
        default_joint_angles = {
            "01_r_hip_pitch_joint": 0,
            "02_r_hip_roll_joint": 0,
            "03_r_thigh_joint": 0,
            "04_r_calf_joint": 0,
            "05_r_ankle_pitch_joint": 0,
            "06_r_ankle_roll_joint": 0,
            "07_l_hip_pitch_joint": 0,
            "08_l_hip_roll_joint": 0,
            "09_l_thigh_joint": 0,
            "10_l_calf_joint": 0,
            "11_l_ankle_pitch_joint": 0,
            "12_l_ankle_roll_joint": 0
        }

    class control(BaseEnvCfg.control):
        control_type = 'P'  # P: position, V: velocity, T: torques
        stiffness = {
            "hip_pitch_joint": 40.0,
            "hip_roll_joint": 20.0,
            "thigh_joint": 20.0,
            "calf_joint": 40.0,
            "ankle_pitch_joint": 40.0,
            "ankle_roll_joint": 20.0,
        }
        damping = {
            "hip_pitch_joint": 0.6,
            "hip_roll_joint": 0.4,
            "thigh_joint": 0.4,
            "calf_joint": 0.6,
            "ankle_pitch_joint": 0.6,
            "ankle_roll_joint": 0.4,
        }

        # action scale: target angle = actionScale * action + defaultAngle
        scale_joint_target = 0.25
        clip_joint_target = 100.
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(BaseEnvCfg.asset):
        file = '{ROOT_DIR}/resources/robots/pi_12dof/pi_12dof.urdf'
        terminate_after_contacts_on = ["base_link"]

        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        ee_offsets = {
            "l_ankle_roll_link": [0.0, 0.0, -0.01],
            "r_ankle_roll_link": [0.0, 0.0, -0.01],
        }
        collapse_fixed_joints = True
        flip_visual_attachments = False

    class rewards(BaseEnvCfg.rewards):
        class terms:
            # sigma, tolerance
            joint_targets_rate = [10.0, 0.0]

        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized

    class domain_rand(BaseEnvCfg.domain_rand):
        randomize_friction = False
        friction_range = [0.9, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.1, 0.1]
        push_robots = False
        push_interval_s = 4
        max_push_vel_xyz = 0
        max_push_avel_xyz = 0

    class termination:
        max_base_lin_vel = 10.0
        max_base_ang_vel = 100.0
        max_height = 3.0

    class viewer(BaseEnvCfg.viewer):
        enable_viewer = False
        overview = True
        record_camera_imgs = False

        # Note: if the viewer is disabled, the following parameters are ignored, only the robot will be recorded in
        # the video. If the viewer is enabled, the following parameters are used to configure the viewer.
        vis_flag = ['ref_only']
        ref_pos_b = [2, 2, 1]
        camera_pos_b = [2., 2., 2]
        ref_lookat = [0, 0, 0.5]



class MimicTrainCfg(BaseTrainCfg):
    algorithm_name = 'PPO'

    class policy:
        log_std_init = 0.0
        actor_hidden_dims = [512, 256]
        critic_hidden_dims = [512, 256]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        surrogate_coef = 1
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs * num_steps / num_minibatches
        learning_rate = 1.e-3
        schedule = 'fixed'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        num_steps_per_env = 24  # per iteration
        max_iterations = 5000  # number of policy updates
        normalize_observation = True
        save_interval = 100  # check for potential saves every this many iterations

        record_gif = True  # need to enable env.viewer.record_camera_imgs and run with wandb
        record_gif_interval = 100
        record_iters = 10  # should be int (* num_steps_per_env)

        # logging
        run_name = 'test'  # name of each run
        experiment_name = 'animrl-pi'

        # load and resume
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model

        wandb = True
        wandb_group = "default"
