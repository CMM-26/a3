import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def joint_transform(joint, angle):
    """Return 4x4 transform for a revolute joint rotated by angle (radians).
    Handles joints with axis zero/None by returning origin translation only.
    """
    transform = np.eye(4)
    transform[:3, :3] = R.from_rotvec(joint.axis * angle).as_matrix()
    transform[:3, 3] = joint.origin[:3, 3]
    return transform


def forward_kinematics(robot, base_pos, base_quat, joint_angles):
    """Compute world transforms for links reachable from the base given joint angles.
    joint_angles: {'names': [...], 'values': [...]}
    Returns dict: {link_name: 4x4 transform}
    """
    root = robot.base_link.name
    base_tf = np.eye(4)
    base_tf[:3, 3] = base_pos
    base_tf[:3, :3] = R.from_quat(base_quat).as_matrix()
    frames = {root: base_tf}

    names = joint_angles.get('names', [])
    values = joint_angles.get('values', [])
    angles_dict = dict(zip(names, values))

    for joint in robot._actuated_joints:  # actuated joints are in correct order for fk
        if joint == root:
            continue
        if joint.parent in frames and joint.child not in frames:
            if joint.name in names:
                transform = frames[joint.parent] @ joint_transform(joint, angles_dict[joint.name])
            else:
                transform = frames[joint.parent] @ joint.origin  # just translate
            frames[joint.child] = transform
    return frames


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    mean_limits = limits.mean(axis=1)
    max_range = (limits[:, 1] - limits[:, 0]).max() / 2

    ax.set_xlim(mean_limits[0] - max_range, mean_limits[0] + max_range)
    ax.set_ylim(mean_limits[1] - max_range, mean_limits[1] + max_range)
    ax.set_zlim(mean_limits[2] - max_range, mean_limits[2] + max_range)


def set_ax_limits(ax, limits):
    """Set axis limits from limits = [[xmin,xmax],[ymin,ymax],[zmin,zmax]]."""
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_zlim(limits[2])


limits_pi = [[-1, 1], [-1, 1], [0, 0.5]]
limits_bob = [[-2, 2], [-2, 2], [0, 2]]


# New: create Line3D artists for the robot from a frame dict.
def create_robot_artists(ax, robot, frames, color='bo-', label='policy', limits=limits_pi):
    """
    Create and return line artists for robot segments visible in `frames`.
    Returns (lines, joints_list) where lines is a list of Line3D objects and
    joints_list is a list of (parent_name, child_name) pairs matching each line.
    """
    lines = []
    joints_list = []
    policy_plotted = False

    # do not clear the axes here to let caller control layout; caller may call ax.cla() first
    for joint in robot.joints:
        if joint.parent in frames and joint.child in frames:
            p1 = frames[joint.parent][:3, 3]
            p2 = frames[joint.child][:3, 3]
            if not policy_plotted:
                line_obj, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color, label=label)
                policy_plotted = True
            else:
                line_obj, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color)
            lines.append(line_obj)
            joints_list.append((joint.parent, joint.child))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_ax_limits(ax, limits)
    return lines, joints_list


# New: update existing Line3D artists in-place given a frame dict
def update_robot_artists(lines, joints_list, frames):
    """
    Update the provided Line3D artists to match the transforms in `frames`.
    lines and joints_list must align (as returned by create_robot_artists).
    """
    for line, (parent, child) in zip(lines, joints_list):
        p1 = frames[parent][:3, 3]
        p2 = frames[child][:3, 3]
        # update XY and Z separately for 3D Line2D objects
        line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
        line.set_3d_properties([p1[2], p2[2]])
    return lines


def plot_robot(ax, robot, frames, color='bo-', label='policy', limits=limits_pi):
    """Efficiently plot the robot as a collection of line segments.
    Parameters:
      ax: 3D matplotlib Axes
      robot: robot object with .joints where each joint has .parent and .child link names
      frames: dict(link_name -> 4x4 transform)
      color: color spec (single color applied to all segments)
      label: legend label for the collection
      limits: axis limits passed to set_ax_limits
    """
    # clear previous artists but keep axis labels/limits if desired
    ax.cla()

    policy_plotted = False
    for joint in robot.joints:
        if joint.parent in frames and joint.child in frames:
            p1 = frames[joint.parent][:3, 3]
            p2 = frames[joint.child][:3, 3]
            if not policy_plotted:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color, label=label)
                policy_plotted = True
            else:
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_ax_limits(ax, limits)


def animate_robot(robot, frames_list, ref_frames_list):
    """
    frames_list/ref_frames_list: precomputed list of {link_name: 4x4 transform} dicts.
    This function creates artists once and updates them in-place for each frame.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.cla()
    ax2.cla()
    policy_lines, policy_joints = create_robot_artists(ax1, robot, frames_list[0], 'bo-', 'policy')
    ref_lines, ref_joints = create_robot_artists(ax2, robot, ref_frames_list[0], 'ro-', 'reference')

    def _update_frame(frame_id):
        update_robot_artists(policy_lines, policy_joints, frames_list[frame_id])
        update_robot_artists(ref_lines, ref_joints, ref_frames_list[frame_id])
        return policy_lines + ref_lines

    ani = FuncAnimation(fig, _update_frame, frames=len(frames_list), interval=20, blit=False)
    return ani
