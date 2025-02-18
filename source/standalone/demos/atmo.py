# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate atmo with a learned policy and MPC.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/demos/atmo.py

"""
import os
import argparse
import torch
import numpy as np 
from IPython import embed
import datetime
import scipy
import onnxruntime as ort

"""Import MPC acados implementation"""
from acados_template          import AcadosOcpSolver
from atmo_mpc                 import create_ocp_solver_description
from atmo_parameters          import params_, get_cost_weights

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
 
def plot_data(actions_log, filtered_actions_log, observations_log, times_log, x_ref_log):
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

    # plot the reference positions on axs[2]
    x_ref = np.array(x_ref_log).squeeze()
    axs[2].plot(times_log, x_ref[:, 0], label="X Ref", linestyle="--")
    axs[2].plot(times_log, x_ref[:, 1], label="Y Ref", linestyle="--")
    axs[2].plot(times_log, x_ref[:, 2], label="Z Ref", linestyle="--")


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

def get_mpc_state(robot):
    # tilt angle
    joint0 = robot.find_joints("base_to_arml")[0]
    tilt_angle = robot.data.joint_pos[:, joint0[0]]

    # position
    pos = robot.data.root_link_pos_w
    pos_transformed = torch.tensor([pos[:,0], -pos[:,1], -pos[:,2]], device=args_cli.device)

    # orientation
    quat = robot.data.root_link_quat_w
    quat_transformed = torch.tensor([quat[:,0], quat[:,1], -quat[:,2], -quat[:,3]],device=args_cli.device)
    roll, pitch, yaw = euler_xyz_from_quat(quat_transformed.unsqueeze(dim=0))

    # wrap angles between -pi and pi 
    roll = torch.fmod(roll + np.pi, 2 * np.pi) - np.pi
    pitch = torch.fmod(pitch + np.pi, 2 * np.pi) - np.pi
    yaw = torch.fmod(yaw + np.pi, 2 * np.pi) - np.pi

    # linear velocity
    lin_vel = robot.data.root_com_lin_vel_w
    lin_vel_transformed = torch.tensor([lin_vel[:,0], -lin_vel[:,1], -lin_vel[:,2]], device=args_cli.device)

    # angular velocity
    ang_vel = robot.data.root_com_ang_vel_b
    ang_vel_transformed = torch.tensor([ang_vel[:,0], -ang_vel[:,1], -ang_vel[:,2]], device=args_cli.device)

    state = torch.cat(
        [
            pos_transformed,
            yaw,
            pitch,
            roll,
            lin_vel_transformed,
            ang_vel_transformed,
            tilt_angle,  
        ],
        dim=-1,
    )
    return state.cpu().detach().numpy()

def mpc_update(solver,N_horizon,mpc_state,x_ref,u_ref,update_cost=True):
    # get current state
    xcurrent = mpc_state[:-1]
    phicurrent = mpc_state[-1]
    
    # adapt cost function
    if update_cost:
        Q_, R_, Qt_ = get_cost_weights(xcurrent[2],phicurrent)
        for j in range(N_horizon):
            solver.cost_set(j, "W", scipy.linalg.block_diag(Q_, R_))
        solver.cost_set(N_horizon, "W", Qt_)

    # set initial state constraint
    solver.set(0, "lbx", xcurrent)
    solver.set(0, "ubx", xcurrent)
    
    # solve ocp
    mpc_status = solver.solve()  
    print(f"mpc_status: {mpc_status}")

    # get first input
    u_opt = solver.get(0, "u")   

    # set the reference and parameters
    for j in range(N_horizon):
        yref = np.hstack((x_ref,u_ref))
        solver.set(j, "yref", yref)
        solver.set(j, "p", np.array([phicurrent]))

    return u_opt

def traj_descent_time(t):
    tilt_vel = 1.0
    descent_vel = 0.5
    z0 = -2.0
    z = np.clip(z0 + descent_vel*t, -2.0, 0.0)
    dz = descent_vel
    x_ref = np.zeros(12)
    x_ref[2] = z
    x_ref[8] = dz
    u_ref = np.zeros(4)
    return x_ref,u_ref,tilt_vel

"""Main function."""
def main():   
    # get parameters
    rl_path                     = params_.get('rl_path')
    N_horizon                   = params_.get('N_horizon')
    acados_ocp_path             = params_.get('acados_ocp_path')
    build_mpc                   = params_.get('build_mpc')
    v_max_absolute              = params_.get('v_max_absolute')
    use_rl                      = params_.get('use_rl')
    disturb                     = params_.get('disturb')
    quantize_tilt_actions       = params_.get('quantize_tilt_actions')
    sim_dt                      = params_.get('sim_dt')
    decimation                  = params_.get('decimation')
    action_update_rate          = params_.get('action_update_rate')
    sim_time                    = params_.get('sim_time')
    sim_steps                   = params_.get('sim_steps')
    alpha                       = params_.get('alpha')
    disturbance_force_scale     = params_.get('disturbance_force_scale')
    disturbance_moment_scale    = params_.get('disturbance_moment_scale')
    desired_position            = params_.get('desired_position')
    initial_pose                = params_.get('initial_pose')
    initial_twist               = params_.get('initial_twist')
    kT                          = params_.get('kT')
    kM                          = params_.get('kM')
    thruster_effectiveness      = params_.get('thruster_effectiveness')

    # get rl model from rl_path
    rl = ort.InferenceSession(rl_path)

    # get run_name
    run_name = "rl" if use_rl else "mpc"
    run_name = run_name

    # define the MPC solver
    ocp  = create_ocp_solver_description()
    acados_ocp_solver = AcadosOcpSolver(
        ocp, 
        json_file=os.path.join(acados_ocp_path, ocp.model.name + '_acados_ocp.json'),
        generate =build_mpc,
        build    =build_mpc
    )

    # initialize the solver
    for stage in range(N_horizon + 1):
        acados_ocp_solver.set(stage, "x", np.zeros(12))
        acados_ocp_solver.set(stage, "p", np.array([0.0]))
    for stage in range(N_horizon):
        acados_ocp_solver.set(stage, "u", np.zeros(4))

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
    actions = torch.ones(robot.num_instances, 5, device=args_cli.device)
    filtered_actions = torch.zeros(robot.num_instances, 5, device=args_cli.device)

    # action history
    action_history_length = 10
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
    x_ref_log            = []

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
            robot.write_root_com_velocity_to_sim(initial_twist)
            robot.reset()

            # reset command
            print(">>>>>>>> Reset!")

        if count % action_update_rate == 0:
            action_history     = torch.cat([actions.clone().unsqueeze(dim=1), action_history[:, :-1]], dim=1)
        obs = get_observations(robot, desired_position, action_history).cpu().detach().numpy()
        
        if count % decimation == 0:
            if use_rl:
                x_ref = np.zeros(12)
                outputs = rl.run(None, {"obs": obs.astype(np.float32)})
                mu = outputs[0]
                actions = torch.tensor(mu, device=args_cli.device)
            else:
                x_ref,u_ref,tilt_vel = traj_descent_time(sim_time)
                mpc_state = get_mpc_state(robot)
                actions = mpc_update(acados_ocp_solver, N_horizon, mpc_state, x_ref, u_ref)
                actions = np.hstack((actions, np.array([tilt_vel])))
                actions = torch.tensor(actions, device=args_cli.device).unsqueeze(0)

        # log observations actions and times
        observations_log.append(obs)
        times_log.append(sim_time)
        actions_log.append(actions.cpu().detach().numpy())
        x_ref_log.append(x_ref)

        # Apply low-pass filter
        filtered_actions[:, :4] = low_pass_filter(actions[:, :4], filtered_actions[:, :4], alpha)
        filtered_actions[:, 4]  = actions[:, 4]

        # log the filtered actions
        filtered_actions_log.append(filtered_actions.cpu().detach().numpy())

        # get the tilt action
        if quantize_tilt_actions: tilt_action = torch.round(filtered_actions[:, 4])
        else: tilt_action     = filtered_actions[:, 4]

        # Update the joint position
        joint_pos[:, 0] = joint_pos[:, 0] + v_max_absolute * tilt_action * sim_dt
        joint_pos       = torch.clamp(joint_pos,0.0,torch.pi/2)
        joint_vel[:, 0] = v_max_absolute * tilt_action

        # Assign the thrust to each of the rotors
        thrust[:, 0, 2] = thruster_effectiveness[0] * kT * filtered_actions[:, 0]
        thrust[:, 1, 2] = thruster_effectiveness[1] * kT * filtered_actions[:, 1]
        thrust[:, 2, 2] = thruster_effectiveness[2] * kT * filtered_actions[:, 2]
        thrust[:, 3, 2] = thruster_effectiveness[3] * kT * filtered_actions[:, 3]

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
        rotor0_pos = robot.data.body_pos_w[:, rotor0].squeeze(0)
        marker_locations = torch.stack([
                                        robot.data.root_link_pos_w, 
                                        desired_position.unsqueeze(0),
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
        count    += 1

        # update buffers
        robot.update(sim_dt)

    return times_log, observations_log, actions_log, filtered_actions_log, x_ref_log, run_name

if __name__ == "__main__":
    # run the main function
    times_log, observations_log, actions_log, filtered_actions_log, x_ref_log, run_name = main()

    # plot the data
    plot_data(actions_log, filtered_actions_log, observations_log, times_log, x_ref_log)

    import os
    import h5py
    import yaml
    import numpy as np
    import torch
    from datetime import datetime

    # Generate a timestamped folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f"/home/m4pc/src/IsaacLab/source/standalone/demos/data/run_{run_name}_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    # Save data in an HDF5 file inside the timestamped folder
    h5_path = os.path.join(run_folder, f"data_{run_name}.h5")
    with h5py.File(h5_path, 'w') as file:
        file.create_dataset("times", data=times_log)
        file.create_dataset("observations", data=observations_log)
        file.create_dataset("actions", data=actions_log)
        file.create_dataset("filtered_actions", data=filtered_actions_log)
        file.create_dataset("x_ref", data=x_ref_log)

    # Convert NumPy/PyTorch objects before saving YAML
    def convert_tensors_and_arrays(obj):
        """Recursively convert NumPy arrays, NumPy scalars, and PyTorch tensors to native types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):  # Handle NumPy scalars
            return obj.item()
        elif isinstance(obj, dict):
            return {key: convert_tensors_and_arrays(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_and_arrays(item) for item in obj]
        return obj

    params_serialized = convert_tensors_and_arrays(params_)

    # Save parameters in a YAML file inside the timestamped folder
    yaml_path = os.path.join(run_folder, f"params_{run_name}.yaml")
    with open(yaml_path, 'w') as file:
        yaml.dump(params_serialized, file)

    print(f"Data and parameters saved in {run_folder}")

    # close sim app
    simulation_app.close()
