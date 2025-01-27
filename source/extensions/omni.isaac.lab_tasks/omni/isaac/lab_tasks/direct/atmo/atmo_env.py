# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
from gymnasium.spaces import Box
import torch
from numpy import pi, deg2rad, exp 

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.envs.ui import BaseEnvWindow
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, quat_from_euler_xyz, axis_angle_from_quat, quat_rotate
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
    # (train: randomize: True, terminate: True, disturb: True, noise: True, actuator_dynamics: True)
    # (test: randomize: False, terminate: False, disturb: True/False, noise True/False, actuator_dynamics: True/False)
    randomize                      = True
    terminate                      = False   
    disturb                        = False 
    noise                          = True 
    actuator_dynamics              = True
    randomize_motor_dynamics       = False
    quantize_tilt_action           = False
    curriculum_update_rate         = 1e4
    curriculum_steps_to_completion = curriculum_update_rate * 20

    # action history
    action_history_length = 1   

    # env
    episode_length_s                         = 5.0    # best was with 8.0 seconds
    sim_dt                                   = 1/100  # training 1/100
    decimation                               = 2      # training 2
    action_space                             = 5
    observation_space                        = 14 + action_space * action_history_length
    num_privileged_obs                       = 10 + action_space 
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
    too_fast_vel_initial               = 2.0   # 2.0 (no curriculum) has given best results so far
    too_fast_vel_final                 = 2.0
    termination_yaw_change             = 1.56

    # acceptance state radius
    delta_d_initial                    = 0.20   # 0.2 has given best results so far (no curriculum)
    delta_d_final                      = 0.20

    # desired velocity
    vx_des, vy_des, vz_des             = 0.0, 0.0, -0.5 
    
    # reward scales
    lin_vel_reward_scale               = 0.12 
    ang_vel_reward_scale               = 0.10
    ori_reward_scale                   = 0.12
    yaw_reward_scale                   = 0.12
    distance_to_goal_reward_scale      = 0.25
    tilt_reward_scale                  = 0.40 
    action_rate_reward_scale           = 0.01
    contact_in_acceptance_reward_scale = 0.15
    ground_thrust_reward_scale         = 0.13
    too_fast_penalty                   = -1.0
    base_link_contact_penalty          = -1.0
    impulse_penalty                    = -1.0 

    # nominal parameters
    kT_0           = 28.15
    kM_0           = 0.018
    max_tilt_vel_0 = pi / 8

    # random force and torque scales
    disturbance_force_scale            = 4 * kT_0 * 0.01                  # 0.01 for training
    disturbance_moment_scale           = 4 * kT_0 * kM_0 * 0.001          # 0.001 for training

    # randomization parameters
    kT_error_scale                     = 0.2 
    kM_error_scale                     = 0.2 
    max_tilt_vel_error_scale           = 0.2

    # low pass filter constant
    T_m_range                           = [0.01, 0.02]
    T_m_0                               = 0.01
    alpha_0                             = 1.0 - exp(-sim_dt / T_m_0).item()
    alpha_range                         = [1.0 - exp(-sim_dt / T_m_range[1]).item(), 1.0 - exp(-sim_dt / T_m_range[0]).item()]

    # observation noise scales 
    pos_noise_scale                    = 0.005                  # 0.5 cm
    quat_noise_scale                   = 0.005                  # 0.5 percent
    lin_vel_noise_scale                = 0.035                  # 0.035 m/s
    ang_vel_noise_scale                = 0.035                  # 2 deg/s or 0.035 rad/s
    tilt_noise_scale                   = 0.018                  # 2 degrees or 0.18 rad/s

class ATMOEnv(DirectRLEnv):
    cfg: ATMOEnvCfg

    def __init__(self, cfg: ATMOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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

        # Initial yaw 
        self._initial_yaw = torch.zeros(self.num_envs, 1, device=self.device)

        # Total impulse
        self._current_impulse = torch.zeros(self.num_envs, 1, device=self.device)

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

        # curriculum parameters
        self.too_fast_vel      = self.cfg.too_fast_vel_initial
        self.delta_d           = self.cfg.delta_d_initial

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "orientation",
                # "yaw",
                "distance_to_goal",
                "tilt",
                "action_rate_l2",
                "contact_in_acceptance",
                "base_link_contact_penalty",
                "died_penalty",
                "impulse_penalty",
                "ground_thrust",
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

        # Clamp actions to [0, 1]
        self._actions = actions.clone().clamp(0.0, 1.0)
        # self._actions = (self._actions + 1)/2

        # Pass thruster actions through a low pass filter
        if self.cfg.actuator_dynamics:
            self._actions_filtered[:, :4] = self.low_pass_filter(self._actions[:,:4], self._actions_filtered[:,:4], self._alpha)
            self._actions_filtered[:, 4]  = self._actions[:, 4]
        else:
            self._actions_filtered = self._actions

        if self.cfg.quantize_tilt_action:
            self._actions_filtered[:, 4] = torch.round(self._actions_filtered[:, 4])

        # print("self._actions", self._actions_filtered)

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
        tilt_action = self._actions_filtered[:, 4]
        self._joint_pos[:, 0, 0] = self._joint_pos[:, 0, 0] + self.max_tilt_vel[:,0] * tilt_action * self.step_dt
        self._joint_pos = torch.clamp(self._joint_pos,0.0,torch.pi/2)

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

        # Apply the disturbance force and moment to the base link
        if self.cfg.disturb:
            self._robot.set_external_force_and_torque(disturbance_force, disturbance_moment, body_ids=self._base_link)
        
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
        self._action_history   = torch.cat([self._actions.clone().unsqueeze(dim=1), self._action_history[:, 1:]], dim=1)
        relative_pos_w         = self._desired_pos_w - self._robot.data.root_link_pos_w
        tilt_angle             = self._robot.data.joint_pos[:, self._joint0[0]].unsqueeze(dim=1)
        impulse                = self.step_dt * torch.sum(torch.linalg.norm((self._contact_sensor.data.net_forces_w_history[:,1,:,:] - self._contact_sensor.data.net_forces_w_history[:,0,:,:]) * self.step_dt, dim=-1),dim=1).unsqueeze(dim=1) # type: ignore
        self._current_impulse  = impulse 

        noise = torch.cat(
            [
                self.cfg.pos_noise_scale * torch.zeros_like(relative_pos_w).uniform_(-1,1),
                self.cfg.quat_noise_scale * torch.zeros_like(self._robot.data.root_link_quat_w).uniform_(-1,1),
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
                self._robot.data.root_link_quat_w,
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
                # self.too_fast_vel * torch.ones_like(self._alpha),
                # self.delta_d * torch.ones_like(self._alpha),
            ],
            dim=-1,
        )

        # get final observations
        obs_policy = obs + self.cfg.noise * noise
        obs_critic = torch.cat([obs, obs_privileged], dim=-1)

        observations = {"policy": obs_policy, "critic": obs_critic}
        return observations

    def _get_rewards(self) -> torch.Tensor:

        # determine if terminal state has been reached
        died, _ = self._get_dones()
        too_fast = torch.linalg.norm(self._robot.data.root_com_lin_vel_w, dim=1) > self.too_fast_vel
        base_link_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, [self._base_link[0]]].squeeze(dim=1)

        # desired velocity
        lin_vel_des = torch.zeros(3, device=self.device)
        lin_vel_des[0], lin_vel_des[1], lin_vel_des[2] = self.cfg.vx_des, self.cfg.vy_des, self.cfg.vz_des

        # contacts
        current_contact_time                 = self.scene["contact_sensor"].data.current_contact_time[:, [self._arml[0], self._armr[0]]]
        num_contact                          = torch.sum(current_contact_time > 0.0, dim=1)
        current_contacts                     = num_contact > 0
        new_contacts                         = torch.logical_and(torch.logical_xor(current_contacts, self._first_contact),current_contacts)
        new_contact_idx                      = torch.nonzero(new_contacts)
        self._first_contact[new_contact_idx] = new_contacts[new_contact_idx]

        # distance to goal
        distance_to_goal        = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        distance_to_goal_mapped = torch.exp(-distance_to_goal / 0.25)

        # height
        height        = self._robot.data.root_link_pos_w[:, 2]
        height_mapped = torch.exp(-torch.square(height) / 0.25)

        # linear and angular velocity
        lin_vel        = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w - lin_vel_des), dim=1)
        lin_vel_mapped = torch.exp(-lin_vel / 0.25)
        ang_vel        = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        ang_vel_mapped = torch.exp(-ang_vel / 0.25)

        # tilt error
        tilt              = self._robot.data.joint_pos[:, self._joint0[0]]
        tilt_error        = torch.square(tilt - pi/2)
        tilt_error_mapped = torch.exp(-tilt_error / 0.25)

        # orientation
        flat_orientation = torch.abs(1 -  self.quat_axis(self._robot.data.root_link_quat_w, 2)[..., 2])
        flat_orientation_mapped = torch.exp(-flat_orientation / 0.25)

        # yaw
        _,_,yaw = euler_xyz_from_quat(self._robot.data.root_link_quat_w)
        yaw_error = torch.square(yaw - self._initial_yaw.squeeze())
        yaw_error_mapped = torch.exp(-yaw_error / 0.25)

        # landing acceptance state
        in_acceptance_ball = (distance_to_goal - self.delta_d) < 0.0

        # action rate
        action_rate_mapped = torch.exp(-torch.sum(torch.square(self._actions - self._action_history[:,0,:]), dim=1)/0.25)

        # reward low thruster actions that occur after first contact
        ground_thrust = torch.sum(torch.square(self._actions[:,:4]), dim=1) * self._first_contact
        ground_thrust_mapped = torch.exp(-ground_thrust / 0.25)

        rewards = {
            "lin_vel":                    lin_vel_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel":                    ang_vel_mapped * self.cfg.ang_vel_reward_scale * self.step_dt,
            "orientation":                flat_orientation_mapped * self.cfg.ori_reward_scale * self.step_dt,
            # "yaw":                        yaw_error_mapped * self.cfg.yaw_reward_scale * self.step_dt,
            "distance_to_goal":           distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt":                       height_mapped * tilt_error_mapped * self.cfg.tilt_reward_scale * self.step_dt,
            "action_rate_l2":             action_rate_mapped * self.cfg.action_rate_reward_scale * self.step_dt,
            "contact_in_acceptance":      current_contacts * in_acceptance_ball * self.cfg.contact_in_acceptance_reward_scale * self.step_dt,
            "base_link_contact_penalty":  base_link_contact * self.cfg.base_link_contact_penalty,
            "died_penalty":               died * self.cfg.too_fast_penalty,
            "impulse_penalty":            self._current_impulse.squeeze(dim=1) * self.cfg.impulse_penalty,
            "ground_thrust":              ground_thrust_mapped * self.cfg.ground_thrust_reward_scale * self.step_dt
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1                                                                              
        died1 = torch.linalg.norm(self._robot.data.root_com_lin_vel_w, dim=1) > self.too_fast_vel
        _,_,yaw = euler_xyz_from_quat(self._robot.data.root_link_quat_w)
        died2 =  torch.abs(yaw - self._initial_yaw.squeeze()) > self.cfg.termination_yaw_change
        if self.cfg.terminate: died = died1 | died2
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
        # also log curriculum values
        extras["Curriculum/too_fast_vel"]               = self.too_fast_vel
        extras["Curriculum/delta_d"]                    = self.delta_d
        extras["Curriculum/common_step_counter"]        = self.common_step_counter
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # # Generate the curriculum
        # if self.common_step_counter % self.cfg.curriculum_update_rate == 0:
        #     coeff = self.common_step_counter / self.cfg.curriculum_steps_to_completion
        #     self.too_fast_vel = max(self.cfg.too_fast_vel_final, self.cfg.too_fast_vel_initial - coeff * (self.cfg.too_fast_vel_initial - self.cfg.too_fast_vel_final))
        #     self.delta_d      = max(self.cfg.delta_d_final, self.cfg.delta_d_initial - coeff * (self.cfg.delta_d_initial - self.cfg.delta_d_final))

        if self.cfg.randomize:
            self._actions[env_ids] = torch.zeros_like(self._actions[env_ids]) # do not randomize initial actions
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            # Sample new desired positions
            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1.0, 1.0)
            # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2])
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] =  torch.zeros_like(self._desired_pos_w[env_ids, 2])
            
            # Get the default root state of the robot
            default_root_state = self._robot.data.default_root_state[env_ids]

            # Make sure the root positions are adjusted by the terrain origins
            default_root_state[:, :3] += self._terrain.env_origins[env_ids]

            # Randomize the initial height
            default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(0.5, 2.0)

            # Randomize the initial linear velocity 
            default_root_state[:, 7:10] = torch.zeros_like(default_root_state[:, 7:10]).uniform_(-0.1, 0.1)

            # Randomize the initial angular velocity
            default_root_state[:, 10:13] = torch.zeros_like(default_root_state[:, 10:13]).uniform_(-0.1, 0.1)

            # Randomize the initial orientation by sampling a random quaternion
            default_root_state[:, 3:7], self._initial_yaw[env_ids,0] = self.random_quaternion(len(env_ids))

            # Randomize the initial tilt angle 
            joint_pos = self._robot.data.default_joint_pos[env_ids]
            random_tilt = torch.zeros_like(joint_pos[:, self._joint0[0]]).uniform_(0, pi/3)
            joint_pos[:, self._joint0[0]] = random_tilt.clone()
            joint_pos[:, self._joint1[0]] = random_tilt.clone()

            # Also set the initial tilt angle
            self._joint_pos[env_ids,:,:] = joint_pos[:, self._joint0[0]].clone().unsqueeze(dim=1).unsqueeze(dim=1)
            
            # Joint velocities are always initialized to zero 
            joint_vel = self._robot.data.default_joint_vel[env_ids]

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

        else:
            self._actions[env_ids] = torch.zeros_like(self._actions[env_ids])
            self._actions_filtered[env_ids] = torch.zeros_like(self._actions_filtered[env_ids])
            self._action_history[env_ids] = torch.zeros_like(self._action_history[env_ids])

            self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.5,0.5)
            self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
            self._desired_pos_w[env_ids, 2] =  torch.zeros_like(self._desired_pos_w[env_ids, 2])

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

    def low_pass_filter(self, x, y, alpha):
        # x input, and y output
        return alpha * x + (1 - alpha) * y

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