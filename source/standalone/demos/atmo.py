# pyright: reportGeneralTypeIssues=false

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
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import h5py
import yaml

"""Import MPC acados implementation"""
from acados_template          import AcadosOcpSolver
from atmo_mpc                 import create_ocp_solver_description, create_ocp_solver_description_phi
from atmo_parameters          import params_, get_cost_weights
from atmo_brtc                import BRTC

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
from omni.isaac.lab.scene import InteractiveSceneCfg, InteractiveScene
from omni.isaac.lab_assets import ATMO_CFG  # isort:skip
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, matrix_from_quat
from omni.isaac.lab.markers import VisualizationMarkers, VisualizationMarkersCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg

@configclass
class ATMOSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    distant_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = ATMO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
 
def visualize_markers(robot,desired_position,visualizer):
    # Get marker location and orientation
    rotor0 = robot.find_bodies("rotor0")[0][0]
    rotor0_pos = robot.data.body_pos_w[:, rotor0]
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

def plot_data(actions_log, filtered_actions_log, pos_log,euler_log,lin_vel_log,ang_vel_log,tilt_angle_log, times_log, x_ref_log):

    # Plot the data
    fig, axs = plt.subplots(4, 1, figsize=(12, 12))
    fig.suptitle("ATMO Simulation")

    # Plot the actions
    actions = np.array(actions_log).squeeze()
    axs[0].plot(times_log, actions[:, 0, 0], label="Thrust 0")
    axs[0].plot(times_log, actions[:, 0,  1], label="Thrust 1")
    axs[0].plot(times_log, actions[:, 0,  2], label="Thrust 2")
    axs[0].plot(times_log, actions[:, 0,  3], label="Thrust 3")
    axs[0].plot(times_log, actions[:, 0,  4], label="Tilt")
    axs[0].set_title("Actions")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Value")
    axs[0].legend()

    # plot filtered actions
    filtered_actions = np.array(filtered_actions_log).squeeze()
    axs[1].plot(times_log, filtered_actions[:, 0,  0], label="Filtered Thrust 0")
    axs[1].plot(times_log, filtered_actions[:, 0,  1], label="Filtered Thrust 1")
    axs[1].plot(times_log, filtered_actions[:, 0,  2], label="Filtered Thrust 2")
    axs[1].plot(times_log, filtered_actions[:, 0,  3], label="Filtered Thrust 3")
    axs[1].plot(times_log, filtered_actions[:, 0,  4], label="Filtered Tilt")
    axs[1].set_title("Actions")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Value")
    axs[1].legend()

    # # Plot the angular rates
    # angular_rates = np.array(observations_log).squeeze()[:, 15:18]
    # axs[2].plot(times_log, angular_rates[:, 0], label="Angular Rate X")
    # axs[2].plot(times_log, angular_rates[:, 1], label="Angular Rate Y")
    # axs[2].plot(times_log, angular_rates[:, 2], label="Angular Rate Z")
    # axs[2].set_title("Observations")
    # axs[2].set_xlabel("Time")
    # axs[2].set_ylabel("Value")
    # axs[2].legend()


    # Plot the observations
    pos = np.array(pos_log).squeeze()
    axs[2].plot(times_log, pos[:, 0,  0], label="X")
    axs[2].plot(times_log, pos[:, 0,  1], label="Y")
    axs[2].plot(times_log, pos[:, 0,  2], label="Z")
    axs[2].set_title("Observations")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Value")
    axs[2].legend()

    # plot the reference positions on axs[2]
    x_ref = np.array(x_ref_log).squeeze()
    axs[2].plot(times_log, x_ref[:, 0,  0], label="X Ref", linestyle="--")
    axs[2].plot(times_log, -x_ref[:, 0,  1], label="Y Ref", linestyle="--")
    axs[2].plot(times_log, -x_ref[:, 0,  2], label="Z Ref", linestyle="--")

    lin_vel = np.array(lin_vel_log).squeeze()
    axs[3].plot(times_log, lin_vel[:, 0,  0], label="VX")
    axs[3].plot(times_log, lin_vel[:, 0,  1], label="VY")
    axs[3].plot(times_log, lin_vel[:, 0,  2], label="VZ")
    axs[3].set_title("Observations")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Value")
    axs[3].legend()

    # also plot tilt angle
    tilt_angle = np.array(tilt_angle_log).squeeze()
    fig, ax = plt.subplots()
    ax.plot(times_log, tilt_angle)
    ax.set_title("Tilt Angle")
    ax.set_xlabel("Time")
    ax.set_ylabel("Tilt Angle")

def low_pass_filter(x, y, alpha):
    # x input, and y output
    return alpha * x + (1 - alpha) * y

def get_state(robot):
    # tilt angle
    joint0 = robot.find_joints("base_to_arml")[0]
    tilt_angle = robot.data.joint_pos[:, joint0[0]]

    # position
    pos = robot.data.root_link_pos_w

    # orientation
    quat = robot.data.root_link_quat_w
    roll, pitch, yaw = euler_xyz_from_quat(quat)
    roll = torch.fmod(roll + np.pi, 2 * np.pi) - np.pi
    pitch = torch.fmod(pitch + np.pi, 2 * np.pi) - np.pi
    yaw = torch.fmod(yaw + np.pi, 2 * np.pi) - np.pi
    euler = torch.stack([roll, pitch, yaw], dim=-1)

    # linear velocity
    lin_vel = robot.data.root_com_lin_vel_w

    # angular velocity
    ang_vel = robot.data.root_com_ang_vel_b

    return pos, euler, quat, lin_vel, ang_vel, tilt_angle

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
    # get state variables
    pos, _, quat, lin_vel, ang_vel, tilt_angle = get_state(robot)

    # transform them for mpc representation
    pos[:,[1,2]]     = -pos[:,[1,2]]
    quat[:,[2,3]]    = -quat[:,[2,3]]
    lin_vel[:,[1,2]] = -lin_vel[:,[1,2]]
    ang_vel[:,[1,2]] = -ang_vel[:,[1,2]]
    
    roll, pitch, yaw = euler_xyz_from_quat(quat)

    # wrap angles between -pi and pi 
    roll = torch.fmod(roll + np.pi, 2 * np.pi) - np.pi
    pitch = torch.fmod(pitch + np.pi, 2 * np.pi) - np.pi
    yaw = torch.fmod(yaw + np.pi, 2 * np.pi) - np.pi

    state = torch.cat(
        [
            pos,
            yaw.unsqueeze(dim=1),
            pitch.unsqueeze(dim=1),
            roll.unsqueeze(dim=1),
            lin_vel,
            ang_vel,
            tilt_angle.unsqueeze(dim=1),  
        ],
        dim=-1,
    )
    return state.cpu().detach().numpy()

def mpc_update(solver,N_horizon,mpc_state,x_ref,u_ref,update_cost=True,phi=False):
            
    # get current state
    xcurrent = mpc_state[:-1]
    phicurrent = mpc_state[-1]

    # set the reference and parameters
    for j in range(N_horizon):
        yref = np.hstack((x_ref,u_ref))
        solver.set(j, "yref", yref)
        if not phi: solver.set(j, "p", np.array([phicurrent]))

    # adapt cost function
    if update_cost:
        Q_, R_, Qt_ = get_cost_weights(xcurrent[2],phicurrent)
        for j in range(N_horizon):
            solver.cost_set(j, "W", scipy.linalg.block_diag(Q_, R_))
        solver.cost_set(N_horizon, "W", Qt_)

    # set initial state constraint
    if not phi:
        solver.set(0, "lbx", xcurrent)
        solver.set(0, "ubx", xcurrent)
    else:
        solver.set(0, "lbx", mpc_state)
        solver.set(0, "ubx", mpc_state)

    # solve ocp
    mpc_status = solver.solve()  

    # get first input
    u_opt = solver.get(0, "u")   

    # get predicted next state 
    x_next = solver.get(1, "x")     

    # get angular rates
    omega = x_next[9:12]

    # get overall thrust
    c_des = u_opt[0] + u_opt[1] + u_opt[2] + u_opt[3]
    
    return u_opt, c_des, omega, mpc_status

def brtc_update(brtc,robot,c_des,omega_des):
    omega             = robot.data.root_com_ang_vel_b.cpu().detach().numpy().squeeze()
    omega_transformed = np.array([omega[0], -omega[1], -omega[2]])
    phi               = robot.data.joint_pos[:, joint0[0]].cpu().detach().numpy().squeeze()
    actions           = brtc.advance(
                                phi=phi,
                                omega=omega_transformed,
                                omega_des=omega_des,
                                c_des=c_des
                        )  
    return actions

def mpc_update_parallel_multithread(solver_list, N_horizon, mpc_states, x_refs, u_refs, update_cost=True, max_workers=4, phi=False):
    batch_size = len(solver_list)
    u_opt_list = [None] * batch_size
    c_des_list = [None] * batch_size
    omega_list = [None] * batch_size
    status_list = [None] * batch_size
    timing_list = [None] * batch_size  # To store per-solver timings

    # Track total execution time
    total_start_time = time.time()

    # Parallel execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for i in range(batch_size):
            # Wrap mpc_update to capture timing
            def timed_mpc_update(idx=i):
                start_time = time.time()
                u_opt, c_des, omega, status = mpc_update(
                    solver_list[idx], N_horizon, mpc_states[idx], x_refs[idx], u_refs[idx], update_cost, phi
                )
                end_time = time.time()
                return u_opt, c_des, omega, status, end_time - start_time

            futures[executor.submit(timed_mpc_update)] = i

        # Collect results
        for future in as_completed(futures):
            i = futures[future]
            try:
                u_opt, c_des, omega, status, elapsed_time = future.result()
                u_opt_list[i] = u_opt
                c_des_list[i] = c_des
                omega_list[i] = omega
                status_list[i] = status
                timing_list[i] = elapsed_time
            except Exception as e:
                print(f"Solver {i} failed with exception: {e}")
                timing_list[i] = None

    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # Print timing info
    print(f"\nTotal Batch Time: {total_elapsed_time:.4f} seconds")
    print(f"Average Per Solver: {np.nanmean(timing_list):.4f} seconds")
    print(f"Fastest Solver: {np.nanmin(timing_list):.4f} s | Slowest Solver: {np.nanmax(timing_list):.4f} s\n")

    return np.array(u_opt_list), np.array(c_des_list), np.array(omega_list), status_list

def mpc_update_parallel(solver_list, N_horizon, mpc_states, x_refs, u_refs, update_cost=True):
    batch_size      = len(solver_list)
    u_opt_list      = []
    status_list     = []
    c_des_list      = []
    omega_list      = []

    for i in range(batch_size):
        
        solver = solver_list[i]
        mpc_state = mpc_states[i]
        x_ref = x_refs[i]
        u_ref = u_refs[i]

        u_opt, c_des, omega, status = mpc_update(solver,N_horizon,mpc_state,x_ref,u_ref,update_cost=update_cost)
        
        status_list.append(status)
        c_des_list.append(c_des)
        omega_list.append(omega)
        u_opt_list.append(u_opt)

    return np.array(u_opt_list), np.array(c_des_list), np.array(omega_list), status_list
    
def brtc_update_parallel(brtc, robots, c_des_list, omega_des_list):
    batch_size = len(robots)
    actions_list = []

    for i in range(batch_size):
        robot = robots[i]
        c_des = c_des_list[i]
        omega_des = omega_des_list[i]

        omega = robot.data.root_com_ang_vel_b.cpu().detach().numpy().squeeze()
        omega_transformed = np.array([omega[0], -omega[1], -omega[2]])
        phi = robot.data.joint_pos[:, joint0[0]].cpu().detach().numpy().squeeze()

        actions = brtc.advance(
            phi=phi,
            omega=omega_transformed,
            omega_des=omega_des,
            c_des=c_des
        )
        actions_list.append(actions)

    return np.array(actions_list)

def initialize_solvers(num_envs, N_horizon, acados_ocp_path, build_mpc, env_origins, x0, u0, phi=False):
    env_origins = env_origins.cpu().detach().numpy()
    solver_list = []
    for i in range(num_envs):
        if not phi:
            ocp = create_ocp_solver_description()
        else:
            ocp = create_ocp_solver_description_phi()
        solver = AcadosOcpSolver(
            ocp,
            json_file=os.path.join(acados_ocp_path, ocp.model.name + '_acados_ocp.json'),
            generate=build_mpc,
            build=build_mpc
        )

        # set initial state and reference
        x0[0] = env_origins[i,0]
        x0[1] = -env_origins[i,1]
        x0[2] = -env_origins[i,2]
        for stage in range(N_horizon + 1):
            solver.set(stage, "x", x0)
            if not phi: solver.set(stage, "p", np.array([0.0]))
        for stage in range(N_horizon):
            solver.set(stage, "u", u0)

        solver_list.append(solver)

    return solver_list

def phi_ref(z):
    phi_ref_ = np.zeros_like(z)
    mask = (z >= 0.0) & (z < 1.0)
    phi_ref_[mask] = (1.0 - z[mask]) * (np.pi / 2)
    return phi_ref_

def traj_descent_time(t,phi,z,z_star,num_envs,root_position):
    root_position = root_position.cpu().detach().numpy()
    descent_vel = 0.5
    z0 = -root_position[:,2]
    z = np.clip(z0 + descent_vel*t, -2.0, 0.0)
    dz = descent_vel * np.ones_like(z)
    x_ref = np.zeros((num_envs,12))
    x_ref[:,0] = root_position[:,0]
    x_ref[:,1] = -root_position[:,1]
    x_ref[:,2] = z
    x_ref[:,8] = dz
    u_ref = np.zeros((num_envs,4))
    accept_height = np.abs(z) < np.abs(z_star)
    accept = phi < np.deg2rad(50)
    np.logical_or(accept,accept_height,accept)
    tilt_vel = accept * np.ones((num_envs,))
    tilt_vel = np.expand_dims(tilt_vel, axis=1)
    # phi_ref_ = phi_ref(z)
    # phi_ref_ = np.expand_dims(phi_ref_, axis=1)
    phi_ref_ = np.ones_like(z) * np.deg2rad(70)
    phi_ref_ = np.expand_dims(phi_ref_, axis=1)
    return x_ref,u_ref,phi_ref_,tilt_vel

def plot_heatmap(fig,ax,scales,angles,metric,run_folder,run_name,plot_name='impact_heatmap',vmin=0.5,vmax=2.5,cmap='RdYlGn_r'):
    # Normalize metric
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Compute circle coordinates
    x = scales * np.cos(angles)
    y = scales * np.sin(angles)

    # Create grid for heatmap focused within the quarter circle
    grid_resolution = 1000
    theta_grid = np.linspace(0, np.pi / 2, grid_resolution)
    radius_grid = np.linspace(0, np.max(scales), grid_resolution)
    Theta, Radius = np.meshgrid(theta_grid, radius_grid)
    X_grid = Radius * np.cos(Theta)
    Y_grid = Radius * np.sin(Theta)

    # Interpolate final_distance over the grid
    heatmap = griddata((x, y), metric, (X_grid, Y_grid), method='cubic')

    # Plot heatmap background
    c = ax.pcolormesh(X_grid, Y_grid, heatmap, cmap=cmap, norm=norm, shading='auto', alpha=0.8, rasterized=True)

    # Create scatter plot
    _ = ax.scatter(x, y, c=metric.flatten(), cmap=cmap, norm=norm, s=50, edgecolor='k')

    # Add colorbar
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=14)

    # Plot quarter-circle arcs for force magnitude
    theta = np.linspace(0, np.pi / 2, 100)  # First quadrant angles
    for radius in scales:
        x_arc = radius * np.cos(theta)
        y_arc = radius * np.sin(theta)
        ax.plot(x_arc, y_arc, color='gray', linestyle='--', linewidth=1)
        # Label the arc at 45 degrees
        label_x = (radius / np.sqrt(2))
        label_y = (radius / np.sqrt(2))
        # ax.text(label_x, label_y, f'{radius:.2f}', fontsize=14, ha='center', va='center')

    # Format plot
    ax.set_aspect('equal')
    ax.set_xlim(0, np.max(scales))
    ax.set_ylim(0, np.max(scales))
    ax.grid(False)

    plt.tight_layout()
    fname = os.path.join(run_folder, f"{plot_name}_{run_name}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')

"""Main function."""
def main():   
    # get parameters
    num_envs                    = params_.get('num_envs')
    mass_deviations             = params_.get('mass_deviations')
    spin_direction              = params_.get('spin_direction')
    vis_markers                 = params_.get('vis_markers')
    rl_path                     = params_.get('rl_path')
    N_horizon                   = params_.get('N_horizon')
    acados_ocp_path             = params_.get('acados_ocp_path')
    build_mpc                   = params_.get('build_mpc')
    v_max_absolute              = params_.get('v_max_absolute')
    control_algorithm           = params_.get('control_algorithm')
    disturb                     = params_.get('disturb')
    disturbance_type            = params_.get('disturbance_type')
    quantize_tilt_actions       = params_.get('quantize_tilt_actions')
    sim_dt                      = params_.get('sim_dt')
    decimation                  = params_.get('decimation')
    action_update_rate          = params_.get('action_update_rate')
    observation_delay           = params_.get('observation_delay')
    sim_time                    = params_.get('sim_time')
    sim_steps                   = params_.get('sim_steps')
    alpha                       = params_.get('alpha')
    white_force_scale           = params_.get('white_force_scale')
    white_moment_scale          = params_.get('white_moment_scale')
    push_time                   = params_.get('push_time')
    push_duration               = params_.get('push_duration')
    desired_position            = params_.get('desired_position')
    initial_pose                = params_.get('initial_pose')
    initial_twist               = params_.get('initial_twist')
    kT                          = params_.get('kT')
    kM                          = params_.get('kM')
    thruster_effectiveness      = params_.get('thruster_effectiveness')
    z_star                      = params_.get('z_star')

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, device=args_cli.device, gravity=(0.0, 0.0, -9.81))
    sim = SimulationContext(sim_cfg)
    
    # Set main camera
    sim.set_camera_view(eye=[5.0, 2.5, 2.5], target=[0.0, 0.0, 0.75])

    # Setup the scene
    scene_cfg = ATMOSceneCfg(num_envs=num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # get robot
    robot = scene["robot"]

    # get rl model from rl_path
    rl = ort.InferenceSession(rl_path)

    # get run_name
    run_name = control_algorithm

    # define the MPC solver for each environment
    if control_algorithm == 'mpc' or control_algorithm == 'mpc-brtc':
        x0 = np.zeros(12)
        u0 = np.zeros(4)
        solvers = initialize_solvers(num_envs, N_horizon, acados_ocp_path, build_mpc, scene.env_origins,x0,u0)

    if control_algorithm == 'mpc-phi':
      x0 = np.zeros(13)
      u0 = np.zeros(5)
      solvers = initialize_solvers(num_envs, N_horizon, acados_ocp_path, build_mpc, scene.env_origins,x0,u0,phi=True)

    # initialize BRTC controller
    if control_algorithm == 'rl-brtc' or control_algorithm == 'mpc-brtc' or control_algorithm == 'brtc':
        brtc = BRTC(h=sim_dt)

    # create push force direction with direction ranging from 0 to pi/2 for each environment
    sqrt_num_envs = int(np.sqrt(num_envs))
    push_scale = torch.linspace(4 * params_['kT'] * 0.15, 4 * params_['kT'] * 2.0 , int(num_envs/sqrt_num_envs), device=args_cli.device)
    push_angle = torch.linspace(0.0, np.pi/2, int(num_envs/sqrt_num_envs), device=args_cli.device)
    angles, scales = torch.meshgrid(push_angle, push_scale)
    angles = angles.reshape(num_envs,1)
    scales = scales.reshape(num_envs,1)
    push_direction = torch.cat([torch.cos(angles), torch.sin(angles), torch.zeros_like(angles)],dim=-1).unsqueeze(1)

    # Get bodies
    rotor0    = robot.find_bodies("rotor0")[0][0]
    rotor1    = robot.find_bodies("rotor1")[0][0]
    rotor2    = robot.find_bodies("rotor2")[0][0]
    rotor3    = robot.find_bodies("rotor3")[0][0]
    base_link = robot.find_bodies("base_link")[0][0]
    arml      = robot.find_bodies("arml")[0][0]
    armr      = robot.find_bodies("armr")[0][0]

    # Get joints
    joint0 = robot.find_joints("base_to_arml")[0][0]
    joint1 = robot.find_joints("base_to_armr")[0][0]

    # set mass of arml
    masses              = robot.root_physx_view.get_masses()
    masses[0,base_link] = mass_deviations[0] * masses[0,base_link]
    masses[0,arml]      = mass_deviations[1] * masses[0,arml]
    masses[0,armr]      = mass_deviations[2] * masses[0,armr]
    robot.root_physx_view.set_masses(masses, torch.tensor([base_link,arml,armr,rotor1,rotor2,rotor0,rotor3]))

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
    action_space = 5
    action_history_length = 10
    action_history = torch.zeros(robot.num_instances, action_history_length, action_space, device=args_cli.device)

    # observation buffer
    observation_history = 0
    num_obs = 19 + action_space * action_history_length
    observation_buffer = torch.zeros(robot.num_instances, 5, num_obs, device=args_cli.device)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    # Define visualizer
    if vis_markers: visualizer = define_markers()

    # Log actions and observations and time and plot them when simulation is done
    times_log            = []
    actions_log          = []
    filtered_actions_log = []
    observations_log     = []
    mpc_state_log        = []
    x_ref_log            = []

    pos_log = []
    euler_log = []
    lin_vel_log = []
    ang_vel_log = []
    tilt_angle_log = []
    status_log = []

    # Simulate physics
    while simulation_app.is_running() and count < sim_steps:
        # reset
        if count % sim_steps == 0:
            # reset counters
            sim_time = 0.0
            count = 0

            # reset dof state
            root_state = robot.data.default_root_state.clone()
            root_state[:, :7] = initial_pose
            root_state[:, :3] += scene.env_origins
            root_state[:,7:]   = initial_twist
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # set desired positions
            desired_position = desired_position.repeat(robot.num_instances,1)
            desired_position += scene.env_origins

            # set joint positions 
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # clear internal buffers
            scene.reset()
            print(">>>>>>>> Reset!")

        # get and log state
        pos, euler, _, lin_vel, ang_vel, tilt_angle = get_state(robot)
        pos_log.append(pos.cpu().detach().numpy())
        euler_log.append(euler.cpu().detach().numpy())
        lin_vel_log.append(lin_vel.cpu().detach().numpy())
        ang_vel_log.append(ang_vel.cpu().detach().numpy())
        tilt_angle_log.append(tilt_angle.cpu().detach().numpy())

        x_ref                = np.zeros(12)
        obs                  = np.zeros((robot.num_instances,num_obs))
        if count % decimation == 0:
            match control_algorithm:
                case "rl":
                    # get rl observations at the correct rate       
                    if count % action_update_rate == 0:
                        action_history     = torch.cat([actions.clone().unsqueeze(dim=1), action_history[:, :-1]], dim=1)
                    obs_current        = get_observations(robot, desired_position, action_history)
                    observation_buffer = torch.cat([obs_current.unsqueeze(1),observation_buffer[:, :-1]], dim=1)
                    obs                = torch.reshape(observation_buffer[:, observation_delay:observation_delay + observation_history + 1],(1,-1))
                    obs                = obs.cpu().detach().numpy().reshape(robot.num_instances,-1)
                    outputs              = rl.run(None, {"obs": obs.astype(np.float32)})
                    actions              = outputs[0]
                    actions              = torch.tensor(actions, device=args_cli.device)
                case "mpc":
                    mpc_state                  = get_mpc_state(robot)
                    x_ref,u_ref,_,tilt_vel     = traj_descent_time(sim_time,mpc_state[:,-1],mpc_state[:,2],z_star,num_envs,root_state[:, :3])
                    mpc_state_log.append(mpc_state)
                    actions, _, _, status      = mpc_update_parallel_multithread(solvers, N_horizon, mpc_state, x_ref, u_ref, update_cost=True)
                    status_log.append(status)
                    actions                    = np.hstack((actions,tilt_vel))
                    actions                    = torch.tensor(actions, device=args_cli.device)
                    # if height is lower than 0.3 set actions of that environment to zero
                    print("height ", np.abs(mpc_state[0,2]))
                    for i in range(num_envs):
                        if np.abs(mpc_state[i,2]) < 0.2:
                            actions[i,:-1] = torch.zeros_like(actions[i,:-1])

                    # if status is not zero for any of the environments set the actions of that environment to zero
                    for i in range(num_envs):
                        if status[i] != 0:
                            actions[i,:-1] = torch.zeros_like(actions[i,:-1])
                case "mpc-phi":
                    mpc_state                  = get_mpc_state(robot)
                    x_ref,u_ref,phi_ref,tilt_vel       = traj_descent_time(sim_time,mpc_state[:,-1],mpc_state[:,2],z_star,num_envs,root_state[:, :3])
                    mpc_state_log.append(mpc_state)
                    x_ref = np.hstack((x_ref,phi_ref))
                    u_ref = np.hstack((u_ref,np.array([[0.0]])))
                    actions, _, _, status      = mpc_update_parallel_multithread(solvers, N_horizon, mpc_state, x_ref, u_ref, update_cost=False, phi=True)
                    status_log.append(status)
                    actions                    = np.hstack((actions,tilt_vel))
                    actions                    = torch.tensor(actions, device=args_cli.device)
                case "rl-brtc":
                    x_ref                = np.zeros(12)
                    outputs              = rl.run(None, {"obs": obs.astype(np.float32)})
                    mu                   = outputs[0]
                    actions              = torch.tensor(mu, device=args_cli.device)
                    actions              = brtc_update(brtc,robot,actions[:,0],actions[:,1:4])
                    actions              = np.hstack((actions, np.array([actions[:,-1]]))) + 1e-6
                    actions              = torch.tensor(actions, device=args_cli.device).unsqueeze(0)
                case "mpc-brtc":
                    x_ref,u_ref,tilt_vel     = traj_descent_time(sim_time)
                    mpc_state                = get_mpc_state(robot)
                    c_des, omega_des, status = mpc_update(solver, N_horizon, mpc_state, x_ref, u_ref,update_cost=True,brtc=True)
                    actions                  = brtc_update(brtc,robot,c_des,omega_des)
                    if mpc_state[-1] > np.deg2rad(70): 
                         tilt_vel = 0.0
                    if status != 0: 
                         actions = np.zeros_like(actions)
                    actions                  = np.hstack((actions, np.array([tilt_vel]))) + 1e-6
                    actions                  = torch.tensor(actions, device=args_cli.device).unsqueeze(0)
                case "brtc":
                    x_ref             = np.zeros(12)   # dummy reference
                    omega_des         = np.array([0.0,0.0,0.0])
                    c_des             = 0.0
                    tilt_vel          = 0.0
                    actions           = brtc_update(brtc,robot,c_des,omega_des)
                    actions           = np.hstack((actions, np.array([tilt_vel]))) + 1e-6
                    actions           = torch.tensor(actions, device=args_cli.device).unsqueeze(0)
                case _:
                    print("Invalid control algorithm: select from rl, mpc, rl-brtc, mpc-brtc")
                    exit()

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

        # Assign the thrust to each of the rotors
        thrust[:,:,2] = (thruster_effectiveness * kT * filtered_actions.reshape(robot.num_instances, 1, 5)[:, :, :4]).squeeze()
        moment[:,:,2] = spin_direction * kM * thrust[:, :, 2]

        # get the tilt action
        if quantize_tilt_actions: tilt_action = torch.round(filtered_actions[:, 4])
        else: tilt_action     = filtered_actions[:, 4]
        tilt_action = tilt_action.reshape(robot.num_instances,1)

        # Update the joint position
        joint_pos  = joint_pos + v_max_absolute * tilt_action * sim_dt
        joint_pos  = torch.clamp(joint_pos,0.0,torch.pi/2)
        joint_vel  = v_max_absolute * tilt_action

        # Apply disturbance
        dist_force         = torch.zeros(robot.num_instances, 1, 3, device=args_cli.device)
        dist_moment        = torch.zeros(robot.num_instances, 1, 3, device=args_cli.device)        
        if disturb:
            match disturbance_type:
                case "white":
                    dist_force        = torch.zeros_like(dist_force).uniform_(-white_force_scale, white_force_scale)
                    dist_moment       = torch.zeros_like(dist_moment).uniform_(-white_moment_scale, white_moment_scale)
                case "push":
                    if sim_time > push_time and sim_time < push_time + push_duration:
                        dist_force         = push_direction * scales.unsqueeze(1)
                case _:
                    print("Invalid disturbance type: select from white and push")
                    exit()            

        # Get total force and moment
        total_force  = torch.cat([thrust, dist_force], dim=1)
        total_moment = torch.cat([moment, dist_moment], dim=1)

        # round force down if it is less than 1e-4
        print("joint_vel ", joint_vel)
        print("joint_pos ", joint_pos)

        # Set force, torque, joint velocity and joint position targets
        robot.set_external_force_and_torque(total_force, total_moment, body_ids=[rotor0,rotor1,rotor2,rotor3,base_link])
        robot.set_joint_velocity_target(joint_vel, joint_ids=[joint0,joint1])
        robot.set_joint_position_target(joint_pos, joint_ids=[joint0,joint1])

        # Write data to sim
        scene.write_data_to_sim()

        # Visualize markers
        if vis_markers: visualize_markers(robot,desired_position,visualizer)

        # perform step
        sim.step()

        # update sim-time
        sim_time += sim_dt
        count    += 1

        # update buffers
        scene.update(sim_dt)

    return status_log, scales, angles, times_log, pos_log, euler_log, lin_vel_log, ang_vel_log, tilt_angle_log, observations_log, mpc_state_log, actions_log, filtered_actions_log, x_ref_log, run_name

if __name__ == "__main__":
    # run the main function
    status_log, scales, angles, times_log, pos_log,euler_log,lin_vel_log,ang_vel_log,tilt_angle_log, observations_log, mpc_state_log, actions_log, filtered_actions_log, x_ref_log, run_name = main()

    # Generate a timestamped folder
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = f"/home/m4pc/src/IsaacLab/source/standalone/demos/data/run_{run_name}_{timestamp}"
    os.makedirs(run_folder, exist_ok=True)

    # plot the data
    # plot_data(actions_log, filtered_actions_log, pos_log,euler_log,lin_vel_log,ang_vel_log,tilt_angle_log, times_log, x_ref_log)

    from scipy.interpolate import griddata

    # Enable LaTeX rendering and set font sizes
    plt.rcParams.update({
        'text.usetex': True,
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 16
    })

    # Compute the impact velocity for each environment
    pos = np.array(pos_log).squeeze()
    lin_vel = np.array(lin_vel_log).squeeze()
    status = np.array(status_log).squeeze()
    impact_vel = []
    failed = []
    for i in range(pos.shape[1]):
        idx = np.where(pos[:, i, 2] < 0.3)[0][0]
        # # check where status is zero and if array is empty then not failed, else failed
        # if np.where(status[:, i] != 0)[0].size == 0:
        #     failed.append(0)
        # else:
        #     failed.append(1)
        impact_vel.append(np.linalg.norm(lin_vel[idx, i, :]))
    impact_vel = np.abs(np.array(impact_vel))
    # failed = np.array(failed)

    # Process data
    scales = scales.squeeze().squeeze().cpu().detach().numpy().squeeze()
    angles = angles.squeeze().squeeze().cpu().detach().numpy().squeeze()
    final_pos = pos[-1, :, :]
    initial_pos = pos[0, :, :]
    final_distance = np.linalg.norm(final_pos[:,:2] - initial_pos[:,:2], axis=1)


    # Figure 1: impact vel heatmap
    metric = impact_vel
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_heatmap(fig,ax,scales,angles,metric,run_folder,run_name,plot_name='impact_heatmap',vmin=0.5,vmax=2.5,cmap='RdYlGn_r')

    # Figure 2: final distance heatmap
    metric = final_distance
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_heatmap(fig,ax,scales,angles,metric,run_folder,run_name,plot_name='dist_heatmap',vmin=0.0,vmax=2.0,cmap='coolwarm')

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

    plt.show()
    # close sim app
    simulation_app.close()
