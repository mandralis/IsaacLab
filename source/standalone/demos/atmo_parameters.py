import numpy as np
import torch
from omni.isaac.lab.utils.math import quat_from_euler_xyz

# declare parameter dictionary
params_ = {}

# rl model
params_['rl_path']                  = "/home/m4pc/src/IsaacLab/logs/rl_games/atmo/2025-02-13_01-37-53-best-delays/nn/exported/policy.onnx"

# high level parameters
params_['use_rl']                   = False
params_['build_mpc']                = False
params_['disturb']                  = False 
params_['quantize_tilt_actions']    = False

# simulation parameters
params_['device']                   = 'cuda:0'
params_['sim_dt']                   = 1 / 100  
params_['decimation']               = 1
params_['action_update_rate']       = 2
params_['sim_time']                 = 6.0
params_['sim_steps']                = int(params_['sim_time']/params_['sim_dt'])
params_['kT']                       = 28.15                                                                 
params_['kM']                       = 0.018                                
params_['T_m']                      = 0.15
params_['alpha']                    = 1.0 - np.exp(-params_['sim_dt'] / params_['T_m']).item()
params_['disturbance_force_scale']  = 4 * params_['kT'] * 0.2
params_['disturbance_moment_scale'] = 4 * params_['kT'] * params_['kM'] * 0.2
params_['desired_position']         = torch.tensor([0.0, 0.0, 0.0],device=params_['device'])
params_['initial_pose']             = torch.cat(
                                        [
                                            torch.tensor([0.0, 0.0, 2.0],device=params_['device']),
                                            quat_from_euler_xyz(torch.tensor(0.0),torch.tensor(0.0),torch.tensor(0.0)).to(params_['device']),
                                        ],
                                         dim=-1,
                                    )
params_['initial_twist']            = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0],device=params_['device']) 
params_['thruster_effectiveness']   = [1.0, 1.0, 1.0, 1.0]

# MPC parameters
params_['N_horizon']             = 10
params_['T_horizon']             = 1.2
params_['acados_ocp_path']       = '/home/m4pc/src/IsaacLab/source/standalone/demos/acados_models/'
params_['u_max']                 = 1.0
params_['v_max_absolute']        = (np.pi/2)/4

# Transition parameters
params_['max_tilt_in_flight']    = np.deg2rad(50)
params_['max_tilt_on_land']      = np.deg2rad(85)
params_['l_pivot_wheel']         = 0.261                                                                                 # distance from pivot point to wheel exterior (look at y distance in SDF and add radius of wheel (0.125))
params_['h_bot_pivot']           = 0.094                                                                                 # distance from bottom plate to pivot point (look at z distance in SDF)
params_['varphi_g']              = np.arctan(params_.get('h_bot_pivot')/params_.get('l_pivot_wheel'))                    # angle of pivot point when robot is on ground and wheels just touch the ground
params_['z_ground_base']         = -0.127                                                                                # (exp: -0.113 TBD) height that optitrack registers when robot is on ground with arms at 0 degrees
params_['h_wheel_ground']        = -0.17                                                                                 # distance from wheel to ground when transition begins                                                 
params_['z_star']                = params_.get('h_wheel_ground') + params_.get('z_ground_base') - (params_.get('l_pivot_wheel') * np.sin(params_.get('max_tilt_in_flight')) - params_.get('h_bot_pivot'))                
params_['u_ramp_down_rate']      = 0.15  

# dynamics parameters (use z-down parameters from excel sheet even though urdf uses z-up parameters)
params_['g']                = 9.81                                                                   # gravitational acceleration
params_['m_base']           = 2.33                                                                   # mass of base
params_['m_arm']            = 1.537                                                                  # mass of arm
params_['m_rotor']          = 0.021                                                                  # mass of rotor 
params_['m']                = params_['m_base'] + 2*params_['m_arm'] + 4*params_['m_rotor']          # total mass 
params_['I_base_xx']        = 0.0067
params_['I_base_yy']        = 0.011
params_['I_base_zz']        = 0.0088
params_['I_base_xy']        = -0.000031
params_['I_base_xz']        = 0.00046
params_['I_base_yz']        = 0.00004
params_['I_arm_xx']         = 0.008732
params_['I_arm_yy']         = 0.036926
params_['I_arm_zz']         = 0.043822
params_['I_arm_xy']         = 0.000007
params_['I_arm_xz']         = -0.000012
params_['I_arm_yz']         = 0.000571
params_['I_rotor_xx']       = 0.000022
params_['I_rotor_yy']       = 0.000022
params_['I_rotor_zz']       = 0.000043
params_['r_BA_right_x']     = 0.0066
params_['r_BA_right_y']     = 0.0685
params_['r_BA_right_z']     = -0.021
params_['r_AG_right_x']     = -0.00032
params_['r_AG_right_y']     = 0.16739
params_['r_AG_right_z']     = -0.02495
params_['r_AR1_x']          = 0.16491
params_['r_AR1_y']          = 0.13673
params_['r_AR1_z']          = -0.069563

# cost function parameters (velocity tracking) flight
params_['w_x']        = 10.0  
params_['w_y']        = 10.0    
params_['w_z']        = 20.0    
params_['w_dx']       = 1.0    
params_['w_dy']       = 1.0    
params_['w_dz']       = 3.0    
params_['w_phi']      = 0.1     
params_['w_th']       = 0.1     
params_['w_psi']      = 0.1     
params_['w_ox']       = 3.0     
params_['w_oy']       = 5.0     
params_['w_oz']       = 1.5     
params_['w_u']        = 1.0       
params_['rho']        = 0.1     
params_['gamma']      = 1.0     

# cost function flight
params_['Q_mat']          = np.diag([
                                     params_['w_x'],
                                     params_['w_y'],
                                     params_['w_z'],
                                     params_['w_psi'],
                                     params_['w_th'],
                                     params_['w_phi'],
                                     params_['w_dx'],
                                     params_['w_dy'],
                                     params_['w_dz'],
                                     params_['w_ox'],
                                     params_['w_oy'],
                                     params_['w_oz']
                                     ])
params_['R_mat']          = params_['rho'] * np.diag([params_['w_u'],params_['w_u'],params_['w_u'],params_['w_u']])
params_['Q_mat_terminal'] = params_['gamma'] * params_['Q_mat']

# cost function parameters near ground
params_['w_x_near_ground']        = 0.0   # exp: ?  TBD  
params_['w_y_near_ground']        = 0.0   # exp: ?  TBD
params_['w_z_near_ground']        = 0.0   # exp: ?  TBD
params_['w_dx_near_ground']       = 0.0   # exp: ?  TBD
params_['w_dy_near_ground']       = 0.0   # exp: ?  TBD
params_['w_dz_near_ground']       = 0.0   # exp: ?  TBD
params_['w_phi_near_ground']      = 0.0   # exp: ?  TBD
params_['w_th_near_ground']       = 0.0   # exp: ?  TBD
params_['w_psi_near_ground']      = 0.0   # exp: ?  TBD  
params_['w_ox_near_ground']       = 3.0   # exp: ?  TBD
params_['w_oy_near_ground']       = 5.0   # exp: ?  TBD
params_['w_oz_near_ground']       = 1.5   # exp: ?  TBD
params_['w_u_near_ground']        = 1.0   # exp: ?  TBD
params_['rho_near_ground']        = 3.0   # exp: ?  TBD
params_['gamma_near_ground']      = 1.0   # exp: ?  TBD

# cost function near ground
params_['Q_mat_near_ground']          = np.diag([params_['w_x_near_ground'],params_['w_y_near_ground'],params_['w_z_near_ground'],params_['w_psi_near_ground'],params_['w_th_near_ground'],params_['w_phi_near_ground'],params_['w_dx_near_ground'],params_['w_dy_near_ground'],params_['w_dz_near_ground'],params_['w_ox_near_ground'],params_['w_oy_near_ground'],params_['w_oz_near_ground']])
params_['R_mat_near_ground']          = params_['rho_near_ground'] * np.diag([params_['w_u_near_ground'],params_['w_u_near_ground'],params_['w_u_near_ground'],params_['w_u_near_ground']])
params_['Q_mat_terminal_near_ground'] = params_['gamma_near_ground'] * params_['Q_mat_near_ground']

def z_schedule(z):
    z,zstar,zg = abs(z),abs(params_['z_star']),abs(params_['z_ground_base'])
    if z >= zstar:
        return 1.0
    elif zg <= z <= zstar:
        return (z - zg)/(zstar - zg)
    else:
        return 0.0
    
def get_blend_factor(z,tilt_angle):
    f_z   = z_schedule(z)
    alpha = f_z * np.cos(tilt_angle)
    return alpha

def get_cost_weights(z,tilt_angle):
    alpha = get_blend_factor(z,tilt_angle)
    Q_,R_,Qt_ = np.copy(params_['Q_mat']),np.copy(params_['R_mat']),np.copy(params_['Q_mat_terminal'])
    Q_  = alpha * params_['Q_mat'] + (1-alpha) * params_['Q_mat_near_ground']
    R_  = alpha * params_['R_mat'] + (1-alpha) * params_['R_mat_near_ground']
    Qt_ = alpha * params_['Q_mat_terminal'] + (1-alpha) * params_['Q_mat_terminal_near_ground']
    return Q_,R_,Qt_