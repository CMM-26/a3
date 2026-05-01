from animRL.cfg.mimic.mimic_pi_config import MimicCfg, MimicTrainCfg


class WalkHWDeployCfg(MimicCfg):
    class env(MimicCfg.env):
        num_envs = 4096

        num_actions = 12
        num_observations = 43
        obs_history_len = 5

        episode_length = 350  # episode length

        reference_state_initialization = False  # initialize state from reference data

    class motion_loader(MimicCfg.motion_loader):
        motion_files = '{ROOT_DIR}/resources/datasets/pi/Walk_new.txt'

    class rewards(MimicCfg.rewards):
        class terms:
            # ----------- TODO 1.3: tune the hyperparameters
            # reward_name = [sigma, tolerance]
            joint_targets_rate = [1.0, 0.0]

            track_base_height = [1.0, 0.0]
            track_base_orientation = [1.0, 0.0]
            track_joint_pos = [1.0, 0.0]
            track_base_vel = [1.0, 0.0]
            track_ee_pos = [1.0, 0.0]
            # ----------- End of implementation

    class control(MimicCfg.control):
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

    class domain_rand(MimicCfg.domain_rand):
        # ----------- TODO 3.1: add domain randomization

        randomize_friction = False
        friction_range = [0.9, 1.0]
        randomize_base_mass = False
        added_mass_range = [-0.1, 0.1]
        push_robots = False
        push_interval_s = 4
        max_push_vel_xyz = 0.1
        max_push_avel_xyz = 0.0
        add_action_delay = False
        dynamic_randomization = 0.0
        obs_noise_scale = 0.0

        # ----------- End of implementation
        randomize_init_state = True

    class init_state(MimicCfg.init_state):
        pos = [0.0, 0.0, 0.36]  # x,y,z [m]
        default_joint_angles = {
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
        }

    class asset(MimicCfg.asset):
        file = '{ROOT_DIR}/resources/robots/pi_12dof_260120/urdf/pi_12dof_260120.urdf'
        ee_offsets = {
            "l_ankle_pitch_link": [0.0, 0.0, 0.0],
            "r_ankle_pitch_link": [0.0, 0.0, 0.0],
        }


class WalkHWDeployTrainCfg(MimicTrainCfg):
    algorithm_name = 'PPO'

    class runner(MimicTrainCfg.runner):
        run_name = 'walk-hw-deploy'
        max_iterations = 5000  # number of policy updates

    class algorithm(MimicTrainCfg.algorithm):
        # ----------- TODO 1.3: tune the hyperparameters
        learning_rate = 1.e-3
        schedule = 'fixed'

        entropy_coef = 0.01
        value_loss_coef = 0.5
        clip_param = 0.2
        desired_kl = 0.01

        bootstrap = True
        # ----------- End of implementation

    class policy(MimicTrainCfg.policy):
        # ----------- TODO 1.3: tune the hyperparameters
        log_std_init = 0
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # ----------- End of implementation
