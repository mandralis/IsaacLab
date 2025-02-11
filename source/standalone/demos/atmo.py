# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate atmo with a learned policy.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/atmo.py

"""

import argparse
import torch
import numpy as np 
import onnxruntime as ort
from IPython import embed

"""Launch Isaac Sim Simulator first."""
from omni.isaac.lab.app import AppLauncher

parser = argparse.ArgumentParser(description="This script demonstrates how to simulate ATMO with a learned policy.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab_assets import ATMO_CFG  # isort:skip
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, matrix_from_quat
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.5, 0.5, 0.5),
            ),
            "arrow_x": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.5, 0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
            ),
            "cube": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "cylinder": sim_utils.CylinderCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "cone": sim_utils.ConeCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
            "mesh": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(10.0, 10.0, 10.0),
            ),
            "mesh_recolored": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(10.0, 10.0, 10.0),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.25, 0.0)),
            ),
            "robot_mesh": sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
                scale=(2.0, 2.0, 2.0),
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.0, 0.1, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)
 
def plot_data(actions_log, filtered_actions_log, observations_log, times_log):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # Plot the data
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle("ATMO Simulation")

    # Plot the actions
    actions = np.array(actions_log).squeeze()
    axs[0].plot(times_log, actions[:, 0], label="Thrust 0")
    axs[0].plot(times_log, actions[:, 1], label="Thrust 1")
    axs[0].plot(times_log, actions[:, 2], label="Thrust 2")
    axs[0].plot(times_log, actions[:, 3], label="Thrust 3")
    axs[0].plot(times_log, actions[:, 4], label="Tilt")
    axs[0].set_title("Actions")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Value")
    axs[0].legend()

    # plot filtered actions
    filtered_actions = np.array(filtered_actions_log).squeeze()
    axs[1].plot(times_log, filtered_actions[:, 0], label="Filtered Thrust 0")
    axs[1].plot(times_log, filtered_actions[:, 1], label="Filtered Thrust 1")
    axs[1].plot(times_log, filtered_actions[:, 2], label="Filtered Thrust 2")
    axs[1].plot(times_log, filtered_actions[:, 3], label="Filtered Thrust 3")
    axs[1].plot(times_log, filtered_actions[:, 4], label="Filtered Tilt")
    axs[1].set_title("Actions")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Value")
    axs[1].legend()


    # Plot the observations
    observations = np.array(observations_log).squeeze()
    axs[2].plot(times_log, observations[:, 0], label="rel X")
    axs[2].plot(times_log, observations[:, 1], label="rel Y")
    axs[2].plot(times_log, observations[:, 2], label="rel Z")
    axs[2].set_title("Observations")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Value")
    axs[2].legend()

    axs[3].plot(times_log, observations[:, 12], label="VX")
    axs[3].plot(times_log, observations[:, 13], label="VY")
    axs[3].plot(times_log, observations[:, 14], label="VZ")
    axs[3].set_title("Observations")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Value")
    axs[3].legend()

    # also plot tilt angle
    tilt_angle = np.rad2deg(observations[:, 18])
    fig, ax = plt.subplots()
    ax.plot(times_log, tilt_angle)
    ax.set_title("Tilt Angle")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tilt Angle")

    plt.show()

def low_pass_filter(x, y, alpha):
    # x input, and y output
    return alpha * x + (1 - alpha) * y

def get_observations(robot, desired_pos_w, action_history):
    joint0 = robot.find_joints("base_to_arml")[0]
    relative_pos_w = desired_pos_w - robot.data.root_link_pos_w
    tilt_angle = robot.data.joint_pos[:, joint0[0]].unsqueeze(dim=1)
    # roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)

    rot = matrix_from_quat(robot.data.root_link_quat_w)
    rot_vector = rot.reshape(-1, 9)

    # quat = robot.data.root_link_quat_w
    obs = torch.cat(
        [
            relative_pos_w,
            rot_vector,
            robot.data.root_com_lin_vel_w,
            robot.data.root_com_ang_vel_b,
            tilt_angle,  
            torch.reshape(action_history, (robot.num_instances, -1)),
        ],
        dim=-1,
    )
    return obs

"""Main function."""
def main():
    # Parameters
    # rl                      = ort.InferenceSession("/home/m4pc/src/IsaacLab/logs/rl_games/atmo/2025-02-05_21-38-02-best-rot-mat-moments/nn/exported/policy.onnx")
    rl                      = ort.InferenceSession("/home/m4pc/src/IsaacLab/logs/rl_games/atmo/2025-02-10_19-44-18/nn/exported/policy.onnx")

    sim_dt                   = 1 / 100  
    decimation               = 1
    sim_time                 = 6.0
    sim_steps                = int(sim_time/sim_dt)


    # create initial pose i..e initial position and quaternion
    initial_pose              = torch.tensor([0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0], device=args_cli.device)
    initial_pose[3:]          = quat_from_euler_xyz(torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0))


    pos_d                    = torch.tensor([0.0, 0.0, 0.0], device=args_cli.device)
    initial_yaw              = torch.tensor([0.0], device=args_cli.device)

    max_tilt_vel             = torch.pi / 8
    kT                       = 28.15
    kM                       = 0.018
    T_m                      = 0.1
    alpha                    = 1.0 - np.exp(-sim_dt / T_m).item()
    disturbance_force_scale  = 4 * kT * 0.2
    disturbance_moment_scale = 4 * kT * kM * 0.2

    disturb                  = False 
    quantize_tilt_actions    = False
    
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[5.0, 2.5, 2.5], target=[0.0, 0.0, 0.75])

    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = ATMO_CFG.replace(prim_path="/World/Robot")
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn, translation=initial_pose[:3])

    # Create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Get bodies
    rotor0    = robot.find_bodies("rotor0")[0]
    rotor1    = robot.find_bodies("rotor1")[0]
    rotor2    = robot.find_bodies("rotor2")[0]
    rotor3    = robot.find_bodies("rotor3")[0]
    base_link = robot.find_bodies("base_link")[0]


    print("rotor0 ", rotor0)
    # Get joints
    joint0 = robot.find_joints("base_to_arml")[0]
    joint1 = robot.find_joints("base_to_armr")[0]

    # Define tensors
    thrust = torch.zeros(robot.num_instances, 4, 3, device=args_cli.device)
    moment = torch.zeros(robot.num_instances, 4, 3, device=args_cli.device)

    # initialize joint pos
    joint_pos = robot.data.default_joint_pos
    joint_vel = robot.data.default_joint_vel

    # initialize actions
    actions = 1.0 * torch.ones(robot.num_instances, 5, device=args_cli.device)
    filtered_actions = torch.zeros(robot.num_instances, 5, device=args_cli.device)

    # action history
    action_history_length = 1
    action_history = torch.zeros(robot.num_instances, action_history_length, 5, device=args_cli.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Define visualizer
    visualizer = define_markers()

    # Log actions and observations and time and plot them when simulation is done
    actions_log          = []
    filtered_actions_log = []
    observations_log     = []
    times_log            = []

    # Simulate physics
    while simulation_app.is_running() and count < sim_steps:
        # reset
        if count % sim_steps == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_link_pose_to_sim(initial_pose)
            robot.write_root_com_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")

        action_history     = torch.cat([actions.clone().unsqueeze(dim=1), action_history[:, 1:]], dim=1)
        obs = get_observations(robot, pos_d, action_history).cpu().detach().numpy()
        if count % decimation == 0:
            # get action from rl 
            outputs = rl.run(None, {"obs": obs.astype(np.float32)})
            mu = outputs[0]
            # sigma = np.exp(outputs[1])
            actions = torch.tensor(mu, device=args_cli.device)   
            print(actions)

        # log observations actions and times
        observations_log.append(obs)
        times_log.append(sim_time)
        actions_log.append(actions.cpu().detach().numpy())

        # Apply low-pass filter
        filtered_actions[:, :4] = low_pass_filter(actions[:, :4], filtered_actions[:, :4], alpha)
        filtered_actions[:, 4]  = actions[:, 4]

        filtered_actions_log.append(filtered_actions.cpu().detach().numpy())

        # Assign the joint positions and velocities
        if quantize_tilt_actions:
            tilt_action = torch.round(filtered_actions[:, 4])
        else:
            tilt_action     = filtered_actions[:, 4]
        print("tilt_action ", tilt_action)

        joint_pos[:, 0] = joint_pos[:, 0] + max_tilt_vel * tilt_action * sim_dt
        joint_pos       = torch.clamp(joint_pos,0.0,torch.pi/2)
        joint_vel[:, 0] = max_tilt_vel * tilt_action

        # Assign the thrust to each of the rotors
        thrust[:, 0, 2] = kT * filtered_actions[:, 0]
        thrust[:, 1, 2] = kT * filtered_actions[:, 1]
        thrust[:, 2, 2] = kT * filtered_actions[:, 2]
        thrust[:, 3, 2] = kT * filtered_actions[:, 3]

        # Assign the moments to each of the rotors
        moment[:, 0, 2] = -kM * thrust[:, 0, 2]
        moment[:, 1, 2] = -kM * thrust[:, 1, 2]
        moment[:, 2, 2] =  kM * thrust[:, 2, 2]
        moment[:, 3, 2] =  kM * thrust[:, 3, 2]

        # Apply disturbance
        disturbance_force        = torch.zeros(robot.num_instances, 3, device=args_cli.device).uniform_(-disturbance_force_scale, disturbance_force_scale)
        disturbance_moment       = torch.zeros(robot.num_instances, 3, device=args_cli.device).uniform_(-disturbance_moment_scale, disturbance_moment_scale)
        if disturb:
            robot.set_external_force_and_torque(disturbance_force, disturbance_moment, body_ids=base_link)

        # Apply the thrust and moments to the rotor bodies
        robot.set_external_force_and_torque(thrust[:, 0, :], moment[:, 0, :], body_ids=rotor0)
        robot.set_external_force_and_torque(thrust[:, 1, :], moment[:, 1, :], body_ids=rotor1)
        robot.set_external_force_and_torque(thrust[:, 2, :], moment[:, 2, :], body_ids=rotor2)
        robot.set_external_force_and_torque(thrust[:, 3, :], moment[:, 3, :], body_ids=rotor3)


        # Apply the joint velocity to the joint
        robot.set_joint_velocity_target(joint_vel[:, 0], joint_ids=joint0)
        robot.set_joint_velocity_target(joint_vel[:, 0], joint_ids=joint1)

        # Apply the joint position to the joint
        robot.set_joint_position_target(joint_pos[:, 0], joint_ids=joint0)
        robot.set_joint_position_target(joint_pos[:, 0], joint_ids=joint1)

        # write data to sim
        robot.write_data_to_sim()

        # Get marker location and orientation
        # get rotor0 location
        rotor0_pos = robot.data.body_pos_w[:, rotor0].squeeze(0)


        marker_locations = torch.stack([
                                        robot.data.root_link_pos_w, 
                                        pos_d.unsqueeze(0),
                                        rotor0_pos
                                        ], 
                                    dim=1
                                    ).squeeze()
        
        marker_orientations = torch.stack([
                                            robot.data.root_link_quat_w, 
                                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=args_cli.device).unsqueeze(0), 
                                            torch.tensor([1.0, 0.0, 0.0, 0.0], device=args_cli.device).unsqueeze(0)
                                            ]
                                        ).squeeze()

        # Visualize markers
        visualizer.visualize(marker_locations, marker_orientations, marker_indices=[0,0,2])

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        robot.update(sim_dt)

    return times_log, observations_log, actions_log, filtered_actions_log

if __name__ == "__main__":
    # run the main function
    times_log, observations_log, actions_log, filtered_actions_log = main()

    # plot the data
    plot_data(actions_log, filtered_actions_log, observations_log, times_log)

    # save data
    np.save("times_log.npy", times_log)
    np.save("observations_log.npy", observations_log)
    np.save("actions_log.npy", actions_log)

    # close sim app
    simulation_app.close()
