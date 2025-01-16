# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
from numpy import pi

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
from omni.isaac.lab_assets import ATMO_CFG  # isort: skip
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
    # env
    episode_length_s = 10.0
    decimation = 2
    action_space = 5
    observation_space = 13
    state_space = 0
    debug_vis = True

    ui_window_class_type = ATMOEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ATMO_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    kT = 28.15
    kM = 0.018
    max_tilt_vel = pi/8

    # contact sensor
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", track_air_time=True,
    )

    # reward scales
    lin_vel_reward_scale = 3.0
    ang_vel_reward_scale = 0.7
    ori_reward_scale     = 12.0
    distance_to_goal_reward_scale = 7.0
    tilt_reward_scale    = 20.0

    get_to_goal_reward_scale = 15.0
    impact_in_acceptance_bonus = 400.0
    undesirable_air_time_reward_scale = 5.0

    # acceptance state
    delta_d = 0.2

    # contact_in_acceptance_reward_scale = 20.0
    # tilt_in_acceptance_reward_scale = 20.0
    # flat_orientation_in_acceptance_reward_scale = 20.0
    # lin_vel_in_acceptance_reward_scale = 20.0
    # ang_vel_in_acceptance_reward_scale = 20.0
    # impact_vel_penalty = 100.0

    # impact_reward_scale  = 1.0
    # ground_thrust_reward_scale = 3.0
    # impact_vel_reward_scale = 10.0



class ATMOEnv(DirectRLEnv):
    cfg: ATMOEnvCfg

    def __init__(self, cfg: ATMOEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Thrust and moments applied to the rotor bodies
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
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

        # First contact flag (all environments start with False)
        self._first_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "orientation",
                "distance_to_goal",
                "tilt",

                "get_to_goal",
                "impact_in_acceptance_bonus",
                "undesirable_air_time",

                # "contact_in_acceptance",
                # "tilt_in_acceptance",
                # "flat_orientation_in_acceptance",
                # "lin_vel_in_acceptance",
                # "ang_vel_in_acceptance",
                # "impact_vel_penalty",
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
        # Clamp actions to [-1, 1]
        self._actions = actions.clone().clamp(-1.0, 1.0)

        # Assign the thrust to each of the rotors
        self._thrust0[:, 0, 2] = self.cfg.kT * (self._actions[:, 0] + 1.0) / 2.0
        self._thrust1[:, 0, 2] = self.cfg.kT * (self._actions[:, 1] + 1.0) / 2.0
        self._thrust2[:, 0, 2] = self.cfg.kT * (self._actions[:, 2] + 1.0) / 2.0
        self._thrust3[:, 0, 2] = self.cfg.kT * (self._actions[:, 3] + 1.0) / 2.0

        # Assign the moments to each of the rotors
        self._moment0[:, 0, 2] = -self.cfg.kM * self._thrust0[:, 0, 2]
        self._moment1[:, 0, 2] = -self.cfg.kM * self._thrust1[:, 0, 2]
        self._moment2[:, 0, 2] =  self.cfg.kM * self._thrust2[:, 0, 2]
        self._moment3[:, 0, 2] =  self.cfg.kM * self._thrust3[:, 0, 2]

        # Assign the joint positions and velocities
        tilt_action = (self._actions[:, 4] + 1.0) / 2.0
        self._joint_pos[:, 0, 0] = self._joint_pos[:, 0, 0] + self.cfg.max_tilt_vel * tilt_action * self.step_dt
        self._joint_pos = torch.clamp(self._joint_pos,0.0,torch.pi/2)

        self._joint_vel[:, 0, 0] = self.cfg.max_tilt_vel * tilt_action

    def _apply_action(self):
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

    def _get_observations_cts(self) -> dict:
        # Get position and orientation of ATMO relative to desired position 
        relative_pos_w, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        # Get current tilt angle
        tilt_angle = self._robot.data.joint_pos[:, self._joint0[0]].unsqueeze(dim=1)
        obs = torch.cat(
            [
                relative_pos_w,
                self._robot.data.root_com_quat_w,
                self._robot.data.root_com_lin_vel_w,
                self._robot.data.root_com_ang_vel_b,
                self._robot.data.projected_gravity_b,
                tilt_angle,     # current tilt angle
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations
    
    def _get_observations(self) -> dict:
        # Get position and orientation of ATMO relative to desired position 
        relative_pos_w, _ = subtract_frame_transforms(
            self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        )

        # Get current tilt angle
        tilt_angle = self._robot.data.joint_pos[:, self._joint0[0]].unsqueeze(dim=1)
        obs = torch.cat(
            [
                relative_pos_w,
                self._robot.data.root_com_lin_vel_w,
                self._robot.data.root_com_ang_vel_b,
                self._robot.data.projected_gravity_b,
                tilt_angle,     # current tilt angle
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards_cts(self) -> torch.Tensor:
        
        # find which environments are currently in contact
        current_contact_time = self.scene["contact_sensor"].data.current_contact_time
        num_contact = torch.sum(current_contact_time > 0.0, dim=1)
        current_contacts  = num_contact > 0

        # new contacts
        new_contacts = torch.logical_and(torch.logical_xor(current_contacts, self._first_contact),current_contacts)
        new_contact_idx = torch.nonzero(new_contacts)

        # update the first contact
        self._first_contact[new_contact_idx] = new_contacts[new_contact_idx]

        # get distance to goal
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # get height
        height = self._robot.data.root_link_pos_w[:, 2]
        height_mapped = torch.exp(-torch.square(height) / 0.25)

        # get linear and angular velocity
        lin_vel     = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w), dim=1)
        ang_vel     = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)

        # get tilt error
        tilt = self._robot.data.joint_pos[:, self._joint0[0]]
        tilt_error = torch.square(tilt - pi/2)
        tilt_error_mapped = torch.exp(-tilt_error / 0.25)

        # tilt reward
        tilt_reward = height_mapped * tilt_error_mapped

        # reward low impact velocities and low tilt error on new contacts
        impact_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w) + torch.square(self._robot.data.root_com_ang_vel_w),dim=1)
        impact_vel_mapped = torch.exp(-impact_vel / 0.25)

        # flat orientation
        # eulerXYZ   = euler_xyz_from_quat(self._robot.data.root_link_quat_w)
        flat_orientation = torch.abs(1 -  self.quat_axis(self._robot.data.root_link_quat_w, 2)[..., 2])
        # flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # impact reward 
        impact = (  
                      self.cfg.impact_vel_reward_scale * impact_vel_mapped 
                    + self.cfg.tilt_reward_scale * tilt_error_mapped 
                    - self.cfg.ori_reward_scale * flat_orientation 
                    + self.cfg.distance_to_goal_reward_scale * distance_to_goal_mapped 
                    - self.cfg.ang_vel_reward_scale * ang_vel
            ) * new_contacts
        
        # reward low thruster actions that occur after first contact
        ground_thrust = torch.sum(torch.square(self._actions[:,:4]), dim=1) * self._first_contact
        ground_thrust_mapped = torch.exp(-ground_thrust / 0.25)

        rewards = {
            "lin_vel": -lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": -ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "orientation": -flat_orientation * self.cfg.ori_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt": tilt_reward * self.cfg.tilt_reward_scale * self.step_dt,
            "impact": impact * self.cfg.impact_reward_scale,
            "ground_thrust": ground_thrust_mapped * self.cfg.ground_thrust_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_rewards(self) -> torch.Tensor:

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
        lin_vel        = torch.sum(torch.square(self._robot.data.root_com_lin_vel_w), dim=1)
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

        # air time after first contact
        undesirable_air_time        = self._first_contact * ~current_contacts * self.step_dt
        undesirable_air_time_mapped = torch.exp(-undesirable_air_time / 0.25)

        # landing acceptance state
        in_acceptance_ball = (distance_to_goal - self.cfg.delta_d) < 0.0

        rewards = {
            "lin_vel":                    lin_vel_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel":                    ang_vel_mapped * self.cfg.ang_vel_reward_scale * self.step_dt,
            "orientation":                flat_orientation_mapped * self.cfg.ori_reward_scale * self.step_dt,
            "distance_to_goal":           distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "tilt":                       height_mapped * tilt_error_mapped * self.cfg.tilt_reward_scale * self.step_dt,
            
            "get_to_goal":                in_acceptance_ball * self.cfg.get_to_goal_reward_scale * self.step_dt,
            "impact_in_acceptance_bonus": in_acceptance_ball * new_contacts * self.cfg.impact_in_acceptance_bonus * self.step_dt,
            "undesirable_air_time":       undesirable_air_time_mapped * self.cfg.undesirable_air_time_reward_scale * self.step_dt,

            
            # "contact_in_acceptance": current_contacts * in_acceptance_ball * self.cfg.contact_in_acceptance_reward_scale * self.step_dt,
            # "tilt_in_acceptance": tilt_error_mapped * in_acceptance_ball * self.cfg.tilt_in_acceptance_reward_scale * self.step_dt,
            # "flat_orientation_in_acceptance": - flat_orientation * in_acceptance_ball * self.cfg.flat_orientation_in_acceptance_reward_scale * self.step_dt,
            # "lin_vel_in_acceptance": lin_vel_mapped * in_acceptance_ball * self.cfg.lin_vel_in_acceptance_reward_scale * self.step_dt,
            # "ang_vel_in_acceptance": ang_vel_mapped * in_acceptance_ball * self.cfg.ang_vel_in_acceptance_reward_scale * self.step_dt,
            
            # 
            # "impact_vel_penalty": (lin_vel_mapped + ang_vel_mapped) * new_contacts * self.cfg.impact_vel_penalty * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        eulerXYZ   = euler_xyz_from_quat(self._robot.data.root_link_quat_w)
        tiltage    = torch.abs(1 -  self.quat_axis(self._robot.data.root_link_quat_w, 2)[..., 2])
        died = torch.any(
            torch.cat(
                [
                    # eulerXYZ[0] > 1.5,
                    # eulerXYZ[0] < -1.5, 
                    # eulerXYZ[1] > 1.5, 
                    # eulerXYZ[1] < -1.5,
                    # tiltage > 0.9,
                    self._robot.data.root_link_pos_w[:, 2] > 100.0
                 ]
                )
        )
        return died,time_out

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
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        # Sample new desired positions
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-0.5, 0.5)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] =  torch.zeros_like(self._desired_pos_w[env_ids, 2])
        
        # Reset robot state with randomization

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

        # Randomize the initial orientation by sampling a random quaternion that makes physical sense
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