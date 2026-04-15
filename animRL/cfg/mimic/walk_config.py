from animRL.cfg.mimic.mimic_pi_config import MimicCfg, MimicTrainCfg


class WalkCfg(MimicCfg):
    class env(MimicCfg.env):
        num_envs = 4096

        num_actions = 12
        num_observations = 48

        episode_length = 150  # episode length

        reference_state_initialization = False  # initialize state from reference data

    class motion_loader(MimicCfg.motion_loader):
        motion_files = '{ROOT_DIR}/resources/datasets/pi/Walk.txt'

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


class WalkTrainCfg(MimicTrainCfg):
    algorithm_name = 'PPO'

    class runner(MimicTrainCfg.runner):
        run_name = 'walk'
        max_iterations = 3000  # number of policy updates

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
