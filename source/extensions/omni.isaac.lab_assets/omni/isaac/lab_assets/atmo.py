# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Caltech's ATMO robot."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

##
# Configuration
##

USD_PATH = "/home/m4pc/src/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Robots/Caltech/atmo.usd"

ATMO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "armr_to_rotor0": 200.0,
            "arml_to_rotor1": -200.0,
            "arml_to_rotor2": 200.0,
            "armr_to_rotor3": -200.0,
        },
        # joint_pos={
        #     "base_to_arml": 0.0,
        #     "base_to_armr": 0.0
        # },
    ),
    actuators={
        "dummy": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    # actuators={
    #     "arml_actuator": ImplicitActuatorCfg(
    #         joint_names_expr=["base_to_arml"],
    #         effort_limit=100.0,
    #         velocity_limit=30.0,
    #         stiffness=1.0,
    #         damping=10.0,
    #     ),
    #     "armr_actuator": ImplicitActuatorCfg(
    #         joint_names_expr=["base_to_armr"], 
    #         effort_limit=100.0, 
    #         velocity_limit=30.0, 
    #         stiffness=1.0, 
    #         damping=10.0
    #     ),
    },
)
"""Configuration for Caltech's ATMO."""
