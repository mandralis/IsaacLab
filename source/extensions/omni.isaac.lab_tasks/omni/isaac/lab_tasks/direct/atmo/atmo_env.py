# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from numpy import pi, exp, copy

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, quat_rotate, matrix_from_quat
from omni.isaac.lab.sensors import ContactSensorCfg, ContactSensor

##
# Pre-defined configs
##
from omni.isaac.lab_assets import ATMO_CFG            # isort: skip
from omni.isaac.lab.markers import CUBOID_MARKER_CFG  # isort: skip

class ATMOEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: ATMOEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class ATMOEnvCfg(DirectRLEnvCfg):
    # high level flags 
    randomize                      = True
    terminate                      = True   
    disturb                        = True 
    noise                          = True 
    actuator_dynamics              = True
    randomize_motor_dynamics       = False
    quantize_tilt_action           = False
    curriculum_update_rate         = 8e3
    curriculum_steps_to_completion = curriculum_update_rate * 10

    # action history
    action_history_length = 10      
    observation_history_length = 0  

    # env
    episode_length_s                         = 5.0    
    sim_dt                                   = 1/50  # training 1/100
    decimation                               = 1      # training 2
    action_space                             = 5

    num_obs                                  = 19 + action_space * action_history_length
    observation_space                        = (observation_history_length + 1) * num_obs
    # observation_space                        = num_obs

    num_privileged_obs                       = 12   
    state_space                              = 0
    debug_vis                                = True
    num_envs                                 = 4096

    ui_window_class_type = ATMOEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=sim_dt,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ATMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True, history_length=2
    )

    # termination conditions
    too_fast_vel                       = 2.0   
    termination_dxy                    = 1.50
    termination_height                 = 3.0

    # observation delay (num steps)
    observation_buffer_length          = 5
    observation_delay                  = 1

    # acceptance state radius
    delta_d                            = 0.40

    # desired velocity
    vx_des, vy_des, vz_des             = 0.0, 0.0, -0.50  
    
    # reward scales
    lin_vel_pen_scale                  = -0.10
    ang_vel_pen_scale                  = -0.30  # best so far 0.10
    spin_pen_scale                     = -0.30  
    action_rate_pen_scale              = -0.07   # best so far 0.30
    # action_rate_tilt_pen_scale         = -0.01  # best so far 0.30
    ground_thrust_pen_scale            = -0.13
    orientation_pen_scale              = -0.10

    impulse_pen                        = -1.0
    died_pen                           = -2.0

    distance_to_goal_xy_rew_scale      = 0.30
    descending_rew_scale               = 0.30
    tilt_rew_scale                     = 0.40
    contact_in_acceptance_rew_scale    = 0.40

    # nominal parameters
    kT_0           = 28.15
    kM_0           = 0.018
    max_tilt_vel_0 = pi / 8

    # random force and torque scales
    disturbance_force_scale            = 4 * kT_0 * 0.05         # best 0.05         
    disturbance_moment_scale           = 4 * kT_0 * kM_0 * 0.05  # best 0.05         

    dist_force_cts_scale              = 4 * kT_0 * 0.0  
    dist_moment_cts_scale             = 4 * kT_0 * kM_0 * 0.0

    # randomization parameters
    kT_error_scale                     = 0.2 
    kM_error_scale                     = 0.2 
    max_tilt_vel_error_scale           = 0.2
    initial_height_range               = [1.0, 2.0]
    initial_lin_vel_range              = [-0.1, 0.1]
    initial_ang_vel_range              = [-0.1, 0.1]
    initial_tilt_range                 = [0.0, pi/6]
    initial_tilt_vel_range             = [0.0, 1.0]

    # low pass filter constant
    step_dt                             = sim_dt * decimation
    T_m_range                           = [0.1, 0.2]
    T_m_0                               = 0.15
    alpha_0                             = 1.0 - exp(-step_dt / T_m_0).item()
    alpha_range                         = [1.0 - exp(-step_dt / T_m_range[1]).item(), 1.0 - exp(-step_dt / T_m_range[0]).item()]

    # observation noise scales 
    pos_noise_scale                    = 0.005                  # 0.5 cm
    quat_noise_scale                   = 0.005                  # 0.5 percent
    lin_vel_noise_scale                = 0.035                  # 0.035 m/s
    ang_vel_noise_scale                = 0.035                  # 2 deg/s or 0.035 rad/s
    tilt_noise_scale                   = 0.018                  # 2 degrees or 0.18 rad/s
    roll_noise_scale                   = 0.008                  # 2 degrees or 0.18 rad/s
    pitch_noise_scale                  = 0.008                  # 2 degrees or 0.18 rad/s
    rot_noise_scale                    = 0.005                  # 0.5 percent

class ATMOEnv(DirectRLEnv):
    cfg: ATMOEnvCfg

    def __init__(self, cfg: ATMOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # self._previous_obs = torch.zeros(self.num_envs, self.cfg.num_obs, device=self.device)
        self._observation_buffer = torch.zeros(self.num_envs, self.cfg.observation_buffer_length, self.cfg.num_obs, device=self.device)

        self.box_extent = 0.1
        self.curriculum_update_time = 0
        self.distance_to_goal_epoch_av = 0.0

        self._current_contacts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._in_acceptance_ball = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._in_acceptance_xy = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # desired velocity
        self._lin_vel_des = torch.zeros(3, device=self.device)
        self._lin_vel_des[0], self._lin_vel_des[1], self._lin_vel_des[2] = self.cfg.vx_des, self.cfg.vy_des, self.cfg.vz_des

        # Initialize the action space
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._actions_filtered = torch.zeros_like(self._actions)
        self._previous_actions = torch.zeros_like(self._actions)

        # Record an action history
        self._action_history = torch.zeros(self.num_envs, self.cfg.action_history_length, gym.spaces.flatdim(self.single_action_space), device=self.device)

        # Thrust and moments applied to the rotor bodies
        self._thrust0 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment0 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment1 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust2 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment2 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._thrust3 = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment3 = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # Joint velocities and positions
        self._joint_vel = torch.zeros(self.num_envs, 1, 1, device=self.device)
        self._joint_pos = torch.zeros(self.num_envs, 1, 1, device=self.device)
        
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Total impulse
        self._current_impulse = torch.zeros(self.num_envs, 1, device=self.device)

        # Acceleration
        self._previous_lin_vel_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._acceleration       = torch.zeros(self.num_envs, 3, device=self.device)

        # First contact flag (all environments start with False)
        self._first_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # disturbance force and moment
        self._disturbance_force  = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._disturbance_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        # push time 
        self._push_time = torch.zeros(self.num_envs, device=self.device)
        self._push_duration = torch.zeros(self.num_envs, device=self.device)

        # time in each environment
        self._time_elapsed = torch.zeros(self.num_envs, device=self.device)

        # filter alpha
        self._alpha = self.cfg.alpha_0 * torch.ones(self.num_envs, 1, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel_pen",
                "ang_vel_pen",
                "action_rate_pen",
                # "action_rate_tilt_pen",
                "ground_thrust_penalty",
                "spin_penalty",
                "orientation_pen",
                "died_penalty",
                "impulse_penalty",
                "distance_to_goal_xy_rew",
                "descending_rew",
                "tilt_rew",
                "contact_in_acceptance_rew",
            ]
        }

        # Get specific body indices
        self._base_link = self._robot.find_bodies("base_link")[0]
        self._rotor0    = self._robot.find_bodies("rotor0")[0]
        self._rotor1    = self._robot.find_bodies("rotor1")[0]
        self._rotor2    = self._robot.find_bodies("rotor2")[0]
        self._rotor3    = self._robot.find_bodies("rotor3")[0]

        # Get the joint indices
        self._joint0 = self._robot.find_joints("base_to_arml")[0]
        self._joint1 = self._robot.find_joints("base_to_armr")[0]

        # Get arml and armr indices
        self._arml = self._robot.find_bodies("arml")[0]
        self._armr = self._robot.find_bodies("armr")[0]

        # Get inertial parameters
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # Initialize kT, kM and max_tilt_vel_0
        self.kT           = self.cfg.kT_0 * torch.ones(self.num_envs, 1, device=self.device)
        self.kM           = self.cfg.kM_0 * torch.ones(self.num_envs, 1, device=self.device)
        self.max_tilt_vel = self.cfg.max_tilt_vel_0 * torch.ones(self.num_envs, 1, device=self.device)

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):

        # increment elapsed time
        self._time_elapsed += self.step_dt

        # Clamp actions to [0, 1] even though sigmoid already achieves this
        self._actions = actions.clone().clamp(0.0, 1.0)

        # Pass thruster actions through a low pass filter
        if self.cfg.actuator_dynamics:
            self._actions_filtered[:, :4] = self._alpha * self._actions[:,:4] + (1 - self._alpha) * self._actions_filtered[:,:4]
            self._actions_filtered[:, 4]  = self._actions[:, 4]
        else:
            self._actions_filtered = self._actions

        if self.cfg.quantize_tilt_action:
            self._actions_filtered[:, 4] = torch.round(self._actions_filtered[:, 4])

        # Assign the thrust to each of the rotors
        self._thrust0[:, 0, 2] = self.kT[:,0] * self._actions_filtered[:, 0]
        self._thrust1[:, 0, 2] = self.kT[:,0] * self._actions_filtered[:, 1]
        self._thrust2[:, 0, 2] = self.kT[:,0] * self._actions_filtered[:, 2]
        self._thrust3[:, 0, 2] = self.kT[:,0] * self._actions_filtered[:, 3]

        # Assign the moments to each of the rotors
        self._moment0[:, 0, 2] = -self.kM[:,0]  * self._thrust0[:, 0, 2]
        self._moment1[:, 0, 2] = -self.kM[:,0]  * self._thrust1[:, 0, 2]
        self._moment2[:, 0, 2] =  self.kM[:,0]  * self._thrust2[:, 0, 2]
        self._moment3[:, 0, 2] =  self.kM[:,0]  * self._thrust3[:, 0, 2]

        # Assign the joint positions and velocities
        tilt_action              = self._actions_filtered[:, 4]
        self._joint_pos[:, 0, 0] = self._joint_pos[:, 0, 0] + self.max_tilt_vel[:,0] * tilt_action * self.physics_dt
        self._joint_pos          = torch.clamp(self._joint_pos,0.0,torch.pi/2)

        self._joint_vel[:, 0, 0] = self.max_tilt_vel[:,0] * tilt_action

    def _apply_action(self):
        
        # Determine whether to push robot
        push = torch.logical_and(self._time_elapsed >= self._push_time, self._time_elapsed <= self._push_time + self._push_duration)
        
        # Compute disturbance force
        disturbance_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        disturbance_force[:,0,0] = self._disturbance_force[:,0,0] * push
        disturbance_force[:,0,1] = self._disturbance_force[:,0,1] * push
        disturbance_force[:,0,2] = self._disturbance_force[:,0,2] * push

        # Compute disturbance moment
        disturbance_moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        disturbance_moment[:,0,0] = self._disturbance_moment[:,0,0] * push
        disturbance_moment[:,0,1] = self._disturbance_moment[:,0,1] * push
        disturbance_moment[:,0,2] = self._disturbance_moment[:,0,2] * push

        # Apply another disturbance force and moment of different nature
        dist_force_cts        = torch.zeros(self.num_envs, 1, 3, device=self.device).uniform_(-self.cfg.dist_force_cts_scale, self.cfg.dist_force_cts_scale)
        dist_moment_cts       = torch.zeros(self.num_envs, 1, 3, device=self.device).uniform_(-self.cfg.dist_moment_cts_scale, self.cfg.dist_moment_cts_scale)

        # Apply the disturbance force and moment to the base link
        if self.cfg.disturb:
            self._robot.set_external_force_and_torque(disturbance_force + dist_force_cts, disturbance_moment + dist_moment_cts, body_ids=self._base_link)

        # Apply the thrust and moments to the rotor bodies
        self._robot.set_external_force_and_torque(self._thrust0, self._moment0, body_ids=self._rotor0)
        self._robot.set_external_force_and_torque(self._thrust1, self._moment1, body_ids=self._rotor1)
        self._robot.set_external_force_and_torque(self._thrust2, self._moment2, body_ids=self._rotor2)
        self._robot.set_external_force_and_torque(self._thrust3, self._moment3, body_ids=self._rotor3)

        # Apply the joint velocity to the joint
        self._robot.set_joint_velocity_target(self._joint_vel[:, 0], joint_ids=self._joint0)
        self._robot.set_joint_velocity_target(self._joint_vel[:, 0], joint_ids=self._joint1)

        # Apply the joint position to the joint
        self._robot.set_joint_position_target(self._joint_pos[:, 0], joint_ids=self._joint0)
        self._robot.set_joint_position_target(self._joint_pos[:, 0], joint_ids=self._joint1)

    def _get_observations(self) -> dict:
        self._action_history     = torch.cat([self._actions.clone().unsqueeze(dim=1), self._action_history[:, :-1]], dim=1)
        relative_pos_w           = self._desired_pos_w - self._robot.data.root_link_pos_w
        tilt_angle               = self._robot.data.joint_pos[:, self._joint0[0]].unsqueeze(dim=1)
        impulse                  = self.step_dt * torch.sum(torch.linalg.norm((self._contact_sensor.data.net_forces_w_history[:,1,:,:] - self._contact_sensor.data.net_forces_w_history[:,0,:,:]) * self.step_dt, dim=-1),dim=1).unsqueeze(dim=1) # type: ignore
        self._current_impulse    = impulse 
        rot         = matrix_from_quat(self._robot.data.root_link_quat_w)
        rot_vector = rot.reshape(-1, 9)

        noise = torch.cat(
            [
                self.cfg.pos_noise_scale * torch.zeros_like(relative_pos_w).uniform_(-1,1),
                self.cfg.rot_noise_scale * torch.zeros_like(rot_vector).uniform_(-1,1),      
                self.cfg.lin_vel_noise_scale * torch.zeros_like(self._robot.data.root_com_lin_vel_w).uniform_(-1,1),
                self.cfg.ang_vel_noise_scale * torch.zeros_like(self._robot.data.root_com_ang_vel_b).uniform_(-1,1),
                self.cfg.tilt_noise_scale * torch.zeros_like(tilt_angle).uniform_(-1,1),
                torch.reshape(torch.zeros_like(self._action_history), (self.num_envs, -1)),
            ],
            dim=-1,
        )
        obs = torch.cat(
            [
                relative_pos_w,      
                rot_vector,                     
                self._robot.data.root_com_lin_vel_w,                   
                self._robot.data.root_com_ang_vel_b,
                tilt_angle,
                torch.reshape(self._action_history, (self.num_envs, -1)),
            ],
            dim=-1,
        )
        obs_privileged = torch.cat(
            [
                self._disturbance_force[:,0,:],
                self._disturbance_moment[:,0,:],
                self._push_time.unsqueeze(dim=1),
                self._push_duration.unsqueeze(dim=1),
                self._time_elapsed.unsqueeze(dim=1),
                self._current_impulse,
                self._actions_filtered,
                self._alpha,
            ],
            dim=-1,
        )

        # get final observations
        obs_current = obs + self.cfg.noise * noise
        self._observation_buffer = torch.cat([obs_current.unsqueeze(1), self._observation_buffer[:, :-1]], dim=1)
        obs_policy = torch.reshape(self._observation_buffer[:,self.cfg.observation_delay:self.cfg.observation_delay+self.cfg.observation_history_length+1], (self.num_envs, -1))
        
        obs_critic = torch.cat([torch.reshape(self._observation_buffer, (self.num_envs, -1)), obs_privileged], dim=-1)

        observations = {"policy": obs_policy, "critic": obs_critic}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        # determine if terminal state has been reached
        died, _ = self._get_dones()

        # contacts
        current_contact_time                 = self.scene["contact_sensor"].data.current_contact_time[:, [self._arml[0], self._armr[0]]]
        num_contact                          = torch.sum(current_contact_time > 0.0, dim=1)
        self._current_contacts               = num_contact > 0
        new_contacts                         = torch.logical_and(torch.logical_xor(self._current_contacts , self._first_contact),self._current_contacts )
        new_contact_idx                      = torch.nonzero(new_contacts)
        self._first_contact[new_contact_idx] = new_contacts[new_contact_idx]

        # distance to goal
        distance_to_goal_xy     = torch.linalg.norm(self._desired_pos_w[:,:2] - self._robot.data.root_link_pos_w[:,:2], dim=1)
        distance_to_goal_xy_mapped = torch.exp(-distance_to_goal_xy / 0.25)

        # height
        height        = self._robot.data.root_link_pos_w[:, 2]
        height_mapped = torch.exp(-torch.square(height) / 0.25)

        # linear and angular velocity
        lin_vel        = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w), dim=1)
        lin_vel_mapped = torch.exp(-lin_vel / 0.25)
        ang_vel        = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b[:,:2]), dim=1)
        ang_vel_mapped = torch.exp(-ang_vel / 0.25)
        spin_vel = torch.square(self._robot.data.root_com_ang_vel_b[:,2])
        spin_vel_mapped = torch.exp(-spin_vel / 0.25)

        # reward descending z velocity
        descending_error        = torch.square(self._robot.data.root_com_lin_vel_w[:, 2] - self._lin_vel_des[2]) * ~self._first_contact
        descending_error_mapped = torch.exp(-descending_error / 0.25)

        # tilt error
        tilt              = self._robot.data.joint_pos[:, self._joint0[0]]
        tilt_error        = torch.square(tilt - pi/2)
        tilt_error_mapped = torch.exp(-tilt_error / 0.25)

        # landing acceptance state
        self._in_acceptance_xy = (distance_to_goal_xy - self.cfg.delta_d) < 0.0

        # action rate
        action_rate      = torch.sum(torch.square(self._actions - self._action_history[:,0,:]), dim=1)
        # action_rate_tilt = torch.sum(torch.square(self._actions[:,-1].unsqueeze(1)  - self._action_history[:,:,-1]), dim=1)
        # action_rate_mapped = torch.exp(-action_rate/0.25)
        # action_rate_tilt_mapped = torch.exp(-action_rate_tilt/0.25)

        # reward low thruster actions that occur after first contact
        ground_thrust = torch.sum(torch.square(self._actions[:,:4]), dim=1) * self._first_contact
        ground_thrust_mapped = torch.exp(-ground_thrust / 0.25)

        # orientation
        flat_orientation = torch.abs(1 -  self.quat_axis(self._robot.data.root_link_quat_w, 2)[..., 2])
        flat_orientation_mapped = torch.exp(-flat_orientation / 0.25)

        rewards = {
            "lin_vel_pen":                    lin_vel                                            * self.cfg.lin_vel_pen_scale                  * self.step_dt, # policy obs: lin_vel (yes)
            "ang_vel_pen":                    ang_vel                                            * self.cfg.ang_vel_pen_scale                  * self.step_dt, # policy obs: ang_vel (yes)
            "action_rate_pen":                action_rate                                        * self.cfg.action_rate_pen_scale              * self.step_dt, # policy obs: previous action (yes)
            # "action_rate_tilt_pen":           action_rate_tilt                                   * self.cfg.action_rate_tilt_pen_scale         * self.step_dt, # policy obs: previous action (yes)
            "ground_thrust_penalty":          ground_thrust                                      * self.cfg.ground_thrust_pen_scale            * self.step_dt, # critic obs: current_contacts (yes)
            "spin_penalty":                   spin_vel                                           * self.cfg.spin_pen_scale                     * self.step_dt, # critic obs: omegaz
            "orientation_pen":                flat_orientation                                   * self.cfg.orientation_pen_scale              * self.step_dt, # policy obs: root_link_quat_w (yes)
 
            "died_penalty":                   died                                               * self.cfg.died_pen,                                          # no observation
            "impulse_penalty":                self._current_impulse.squeeze(dim=1)               * self.cfg.impulse_pen,                                       # critic obs: current_impulse (yes)
             
            "distance_to_goal_xy_rew":        distance_to_goal_xy_mapped                         * self.cfg.distance_to_goal_xy_rew_scale      * self.step_dt, # policy obs: relative_pos (yes)
            "descending_rew":                 descending_error_mapped                            * self.cfg.descending_rew_scale               * self.step_dt, # no observation
            "tilt_rew":                       tilt_error_mapped                                  * self.cfg.tilt_rew_scale                     * self.step_dt, # policy obs: tilt_angle (yes)
            "contact_in_acceptance_rew":      self._current_contacts  * self._in_acceptance_xy   * self.cfg.contact_in_acceptance_rew_scale    * self.step_dt, # critic obs: in_acceptance_ball (yes)

        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1                                                                              
        died1 = torch.linalg.norm(self._robot.data.root_com_lin_vel_w, dim=1) > self.cfg.too_fast_vel
        died2 =  torch.linalg.norm(self._desired_pos_w[:,:2] - self._robot.data.root_link_pos_w[:,:2], dim=1) > self.cfg.termination_dxy
        died3 =  self._robot.data.root_link_pos_w[:, 2] > self.cfg.termination_height
        if self.cfg.terminate: died = died1 | died2 | died3
        else: died = torch.zeros(self.num_envs, device=self.device)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        extras["Metrics/distance_to_goal_epoch_av"] = self.distance_to_goal_epoch_av
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        if self.cfg.randomize:
            self._observation_buffer[env_ids] = torch.zeros_like(self._observation_buffer[env_ids])

            self._actions[env_ids] = torch.ones_like(self._actions[env_ids])
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            # Sample new desired positions
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-self.box_extent, self.box_extent)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] =  torch.zeros_like(self._desired_pos_w[env_ids, 2])
            
            # Get the default root state of the robot
            default_root_state = self._robot.data.default_root_state[env_ids]

            # Make sure the root positions are adjusted by the terrain origins
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            # Randomize the initial height
            default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(self.cfg.initial_height_range[0], self.cfg.initial_height_range[1])

            # Randomize the initial linear velocity 
            default_root_state[:, 7:10] = torch.zeros_like(default_root_state[:, 7:10]).uniform_(self.cfg.initial_lin_vel_range[0], self.cfg.initial_lin_vel_range[1])

            # Randomize the initial angular velocity
            default_root_state[:, 10:13] = torch.zeros_like(default_root_state[:, 10:13]).uniform_(self.cfg.initial_ang_vel_range[0], self.cfg.initial_ang_vel_range[1])

            # Randomize the initial orientation by sampling a random quaternion
            default_root_state[:, 3:7], _ = self.random_quaternion(len(env_ids))

            # Randomize the initial tilt angle 
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            random_tilt = torch.zeros_like(joint_pos[:, self._joint0[0]]).uniform_(self.cfg.initial_tilt_range[0], self.cfg.initial_tilt_range[1])
            joint_pos[:, self._joint0[0]] = random_tilt.clone()
            joint_pos[:, self._joint1[0]] = random_tilt.clone()

            # Also set the initial tilt angle
            self._joint_pos[env_ids,:,:] = joint_pos[:, self._joint0[0]].clone().unsqueeze(dim=1).unsqueeze(dim=1)
            
            # Joint velocities randomly initialized
            joint_vel = self._robot.data.default_joint_vel[env_ids]
            random_vel = torch.zeros_like(joint_vel[:, self._joint0[0]]).uniform_(self.cfg.initial_tilt_vel_range[0], self.cfg.initial_tilt_vel_range[1])
            joint_vel[:, self._joint0[0]] = random_vel.clone()
            joint_vel[:, self._joint1[0]] = random_vel.clone()

            # Write the root state to the simulation
            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

            # Reset the first contact array 
            self._first_contact[env_ids] = False

            # Randomize kT, kM and max_tilt_vel
            self.kT[env_ids]           = self.cfg.kT_0 * ( 1 + self.cfg.kT_error_scale * torch.zeros_like(self.kT[env_ids]).uniform_(-1., 1.))
            self.kM[env_ids]           = self.cfg.kM_0 * ( 1 + self.cfg.kM_error_scale * torch.zeros_like(self.kM[env_ids]).uniform_(-1., 1.))
            self.max_tilt_vel[env_ids] = self.cfg.max_tilt_vel_0 * ( 1 + self.cfg.max_tilt_vel_error_scale * torch.zeros_like(self.max_tilt_vel[env_ids]).uniform_(-1., 1.))

            # Sample disturbance directions for the episode
            disturbance_force_direction  = torch.normal(0.0, 1.0, size=(self.num_envs,1,3),device=self.device)
            disturbance_force_direction  = disturbance_force_direction / (torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6)
            
            disturbance_moment_direction  = torch.normal(0.0, 1.0, size=(self.num_envs,1,3),device=self.device)
            disturbance_moment_direction  = disturbance_moment_direction / (torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6)

            # Randomize the push time and push duration
            self._push_time[env_ids]     = self.cfg.episode_length_s * torch.zeros_like(self._push_time[env_ids]).uniform_(0.0, 0.8)
            self._push_duration[env_ids] = torch.zeros_like(self._push_duration[env_ids]).uniform_(0.0, 1.0)

            # Randomize disturbance intensity 
            force_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_force_scale)
            moment_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_moment_scale)
            self._disturbance_force  = force_intensity * disturbance_force_direction
            self._disturbance_moment = moment_intensity * disturbance_moment_direction

            # Randomize self._alpha
            if self.cfg.randomize_motor_dynamics:
                self._alpha = torch.zeros_like(self._alpha).uniform_(self.cfg.alpha_range[0], self.cfg.alpha_range[1])

            # Reset time
            self._time_elapsed[env_ids] = 0.0

            # Reset impulse
            self._current_impulse[env_ids] = 0.0

            # Reset current contacts and in acceptance ball
            self._current_contacts = torch.zeros_like(self._current_contacts)
            self._in_acceptance_ball = torch.zeros_like(self._in_acceptance_ball)
            self._in_acceptance_xy = torch.zeros_like(self._in_acceptance_xy)

        else:
            self._observation_buffer[env_ids] = torch.zeros_like(self._observation_buffer[env_ids])

            self._actions[env_ids] = torch.zeros_like(self._actions[env_ids])
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.1,0.1)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2]  =  torch.zeros_like(self._desired_pos_w[env_ids, 2])

            default_root_state = self._robot.data.default_root_state[env_ids]
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            joint_pos = self._robot.data.default_joint_pos[env_ids]
            self._joint_pos[env_ids,:,:] = joint_pos[:, self._joint0[0]].clone().unsqueeze(dim=1).unsqueeze(dim=1)

            joint_vel = self._robot.data.default_joint_vel[env_ids]

            self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)

            # Choose disturbance directions to test robot
            disturbance_force_direction  = torch.normal(0.0, 1.0, size=(1,1,3),device=self.device).repeat(self.num_envs,1,1)
            disturbance_force_direction  = disturbance_force_direction / (torch.linalg.norm(disturbance_force_direction, dim=1).unsqueeze(dim=1) + 1e-6)
            
            disturbance_moment_direction  = torch.normal(0.0, 1.0, size=(1,1,3),device=self.device).repeat(self.num_envs,1,1)      
            disturbance_moment_direction  = disturbance_moment_direction / (torch.linalg.norm(disturbance_moment_direction, dim=1).unsqueeze(dim=1) + 1e-6)

            # Randomize the push time and push duration
            self._push_time[env_ids]     = 0.5 * torch.ones_like(self._push_time[env_ids])
            self._push_duration[env_ids] = 1.0 * torch.ones_like(self._push_duration[env_ids])

            # Randomize disturbance intensity 
            force_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_force_scale)
            moment_intensity = torch.normal(torch.tensor(0.0), self.cfg.disturbance_moment_scale)
            self._disturbance_force  = force_intensity * disturbance_force_direction
            self._disturbance_moment = moment_intensity * disturbance_moment_direction
            
            self._first_contact[env_ids]   = False
            self._time_elapsed[env_ids]    = 0.0
            self._current_impulse[env_ids] = 0.0

            # Reset current contacts and in acceptance ball
            self._current_contacts = torch.zeros_like(self._current_contacts)
            self._in_acceptance_ball = torch.zeros_like(self._in_acceptance_ball)
            self._in_acceptance_xy = torch.zeros_like(self._in_acceptance_xy)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def random_quaternion(self, num):
        """Returns sampled rotation around z-axis.

        Args:
            num: The number of rotations to sample.

        Returns:
            Sampled quaternion in (w, x, y, z). Shape is (num, 4).
        """
        roll = torch.pi / 6 * (2 * torch.rand(num, dtype=torch.float) - 1)
        pitch = torch.pi / 6 * (2 * torch.rand(num, dtype=torch.float) - 1)
        yaw = 2 * torch.pi * torch.rand(num, dtype=torch.float)

        return quat_from_euler_xyz(roll, pitch, yaw), yaw.to(self.device)
    
    def quat_axis(self, q, axis=0):
        # type: (Tensor, int) -> Tensor
        basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
        basis_vec[:, axis] = 1
        return quat_rotate(q, basis_vec)

    def tensor_clamp(self, t, min_t, max_t):
        return torch.max(torch.min(t, max_t), min_t)