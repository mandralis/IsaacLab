# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate atmo.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/atmo.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import torch

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to simulate a quadcopter.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.sim import SimulationContext

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ATMO_CFG  # isort:skip

max_tilt_vel = torch.pi / 8
kT = 28.15
kM = 0.018

def get_observations(robot, desired_pos_w):
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
        ],
        dim=-1,
    )
    return obs


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[5.0, 2.5, 2.5], target=[0.0, 0.0, 0.75])

    # load rl model
    rl = torch.jit.load("/home/m4pc/src/IsaacLab/logs/rsl_rl/atmo/2025-01-16_17-22-04/exported/policy.pt",
                        map_location=args_cli.device)
    # rl = torch.jit.load("/home/m4pc/src/IsaacLab/logs/rsl_rl/atmo/2025-01-18_15-42-02/exported/policy.pt",
                        # map_location=args_cli.device)
    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robots
    robot_cfg = ATMO_CFG.replace(prim_path="/World/Robot")
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

    # create handles for the robots
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

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 2000 == 0:
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
        
        # get action from rl 
        pos_d = torch.tensor([0.0, 0.0, 0.0], device=args_cli.device)
        obs = get_observations(robot, pos_d)
        actions = rl(obs)
        actions = actions.clone().clamp(-1.0, 1.0)
        actions = (actions + 1.0) / 2.0
        print(actions.cpu().detach().numpy())

        # Assign the joint positions and velocities
        tilt_action = actions[:, 4]
        joint_pos[:, 0] = joint_pos[:, 0] + max_tilt_vel * tilt_action * sim_dt
        joint_pos = torch.clamp(joint_pos,0.0,torch.pi/2)
        joint_vel[:, 0] = max_tilt_vel * tilt_action

        # Assign the thrust to each of the rotors
        thrust[:, 0, 2] = kT * actions[:, 0]
        thrust[:, 1, 2] = kT * actions[:, 1]
        thrust[:, 2, 2] = kT * actions[:, 2]
        thrust[:, 3, 2] = kT * actions[:, 3]

        # Assign the moments to each of the rotors
        moment[:, 0, 2] = -kM * thrust[:, 0, 2]
        moment[:, 1, 2] = -kM * thrust[:, 1, 2]
        moment[:, 2, 2] =  kM * thrust[:, 2, 2]
        moment[:, 3, 2] =  kM * thrust[:, 3, 2]

        # add random force disturbance
        disturbance_force_scale = 4 * kT * 0.06
        disturbance_moment_scale = 4 * kT * kM * 0.007
        disturbance_force = torch.zeros(robot.num_instances, 3, device=args_cli.device).uniform_(-disturbance_force_scale, disturbance_force_scale)
        disturbance_moment = torch.zeros(robot.num_instances, 3, device=args_cli.device).uniform_(-disturbance_moment_scale, disturbance_moment_scale)

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

        # Apply disturbance
        robot.set_external_force_and_torque(disturbance_force, disturbance_moment, body_ids=base_link)

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
