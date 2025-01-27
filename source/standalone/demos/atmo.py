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

def low_pass_filter(x, y, alpha):
    # x input, and y output
    return alpha * x + (1 - alpha) * y

def get_observations(robot, desired_pos_w, actions):
    joint0 = robot.find_joints("base_to_arml")[0]
    relative_pos_w = desired_pos_w - robot.data.root_link_pos_w
    tilt_angle = robot.data.joint_pos[:, joint0[0]].unsqueeze(dim=1)
    obs = torch.cat(
        [
            relative_pos_w,
            robot.data.root_link_quat_w,
            robot.data.root_com_lin_vel_w,
            robot.data.root_com_ang_vel_b,
            tilt_angle,  
            actions,
        ],
        dim=-1,
    )
    return obs

"""Main function."""
def main():
    # Parameters
    rl                       = ort.InferenceSession("/home/m4pc/src/IsaacLab/logs/rl_games/atmo/2025-01-25_16-43-12/nn/exported/policy.onnx")

    sim_dt                   = 1 / 200
    decimation               = 2
    sim_time                 = 8.0

    pos_d                    = torch.tensor([0.5, -0.5, 0.0], device=args_cli.device)

    max_tilt_vel             = torch.pi / 8
    kT                       = 28.15
    kM                       = 0.018
    T_m                      = 0.015
    alpha                    = 1.0 - np.exp(-sim_dt / T_m).item()
    disturbance_force_scale  = 4 * kT * 0.25
    disturbance_moment_scale = 4 * kT * kM * 0.25    
    
    # Load kit helper
    sim_steps = int(sim_time/sim_dt)
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
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # Create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Get bodies
    rotor0 = robot.find_bodies("rotor0")[0]
    rotor1 = robot.find_bodies("rotor1")[0]
    rotor2 = robot.find_bodies("rotor2")[0]
    rotor3 = robot.find_bodies("rotor3")[0]
    base_link = robot.find_bodies("base_link")[0]

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
    actions = torch.zeros(robot.num_instances, 5, device=args_cli.device)
    filtered_actions = torch.zeros(robot.num_instances, 5, device=args_cli.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % sim_steps == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_link_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_com_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            # reset command
            print(">>>>>>>> Reset!")

        if count % decimation == 0:
            # get action from rl 
            obs = get_observations(robot, pos_d, actions).cpu().detach().numpy()
            outputs = rl.run(None, {"obs": obs.astype(np.float32)})
            mu = outputs[0]
            # sigma = np.exp(outputs[1])
            actions = torch.tensor(mu, device=args_cli.device)   

        # Apply low-pass filter
        filtered_actions[:, :4] = low_pass_filter(actions[:, :4], filtered_actions[:, :4], alpha)
        filtered_actions[:, 4]  = actions[:, 4]

        # Assign the joint positions and velocities
        tilt_action     = filtered_actions[:, 4]
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

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count += 1

        # update buffers
        robot.update(sim_dt)

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
