import scipy
import numpy as np
from acados_template import AcadosModel, AcadosOcp
from casadi import SX, vertcat, sin, cos, inv, Function, MX
from atmo_parameters import params_

# constraint variables
u_max                       = params_.get('u_max')
v_max_absolute              = params_.get('v_max_absolute')

# collocation parameters in flight
N_horizon                   = params_.get('N_horizon')
T_horizon                   = params_.get('T_horizon')

# cost function in flight
Q_mat                       = params_.get('Q_mat')
R_mat                       = params_.get('R_mat')
Q_mat_terminal              = params_.get('Q_mat_terminal')

# temporary references (these get overwritten the first time mpc is called)
x_ref                       = np.zeros(12,dtype='float')
u_ref                       = np.zeros(4,dtype='float')
x0                          = np.zeros(12,dtype='float')
varphi0                     = 0.0

# export directory
acados_ocp_path             = params_.get('acados_ocp_path')

# get robot model parameters 
gravity              = params_.get('g')                                 
kT                   = params_.get('kT')
kM                   = params_.get('kM')
m_base               = params_.get('m_base')
m_arm                = params_.get('m_arm')
m_rotor              = params_.get('m_rotor')
m                    = params_.get('m')
I_base_xx            = params_.get('I_base_xx')
I_base_yy            = params_.get('I_base_yy')
I_base_zz            = params_.get('I_base_zz')
I_base_xy            = params_.get('I_base_xy')
I_base_xz            = params_.get('I_base_xz')
I_base_yz            = params_.get('I_base_yz')
I_arm_xx             = params_.get('I_arm_xx')
I_arm_yy             = params_.get('I_arm_yy')
I_arm_zz             = params_.get('I_arm_zz')
I_arm_xy             = params_.get('I_arm_xy')
I_arm_xz             = params_.get('I_arm_xz')
I_arm_yz             = params_.get('I_arm_yz')
I_rotor_xx           = params_.get('I_rotor_xx')
I_rotor_yy           = params_.get('I_rotor_yy')
I_rotor_zz           = params_.get('I_rotor_zz')
r_BA_right_x         = params_.get('r_BA_right_x')
r_BA_right_y         = params_.get('r_BA_right_y')
r_BA_right_z         = params_.get('r_BA_right_z')
r_AG_right_x         = params_.get('r_AG_right_x')
r_AG_right_y         = params_.get('r_AG_right_y')
r_AG_right_z         = params_.get('r_AG_right_z')
r_AR1_x              = params_.get('r_AR1_x')
r_AR1_y              = params_.get('r_AR1_y')
r_AR1_z              = params_.get('r_AR1_z')

# individual terms of robot dynamics equation
def M_func():
    x   = SX.sym('x',12)
    phi = SX.sym('phi',1)

    # get state variables
    x_B      = x[0]
    y_B      = x[1]
    z_B      = x[2]
    theta_Bz = x[3]
    theta_By = x[4]
    theta_Bx = x[5]
    dx_B     = x[6]
    dy_B     = x[7]
    dz_B     = x[8]
    omega_Bx = x[9]
    omega_By = x[10]
    omega_Bz = x[11]

    out = SX(
        np.array([
[                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2*m_arm + m_base + 4*m_rotor,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                       2*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By))*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)), 2*m_arm*r_BA_right_z*cos(theta_By)*cos(theta_Bz) - 2*m_arm*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*r_AG_right_x*sin(theta_Bx)*sin(theta_Bz) + 4*m_rotor*r_BA_right_z*cos(theta_By)*cos(theta_Bz) + 2*m_arm*r_AG_right_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 4*m_rotor*r_AR1_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 2*m_arm*r_AG_right_y*cos(theta_By)*cos(theta_Bz)*sin(phi) + 4*m_rotor*r_AR1_y*cos(theta_By)*cos(theta_Bz)*sin(phi) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  -2*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By))*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x)],
[                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    2*m_arm + m_base + 4*m_rotor,                                                                                                                                                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                                                                      -2*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz))*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)), 2*m_arm*r_AG_right_x*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*r_BA_right_z*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 4*m_rotor*r_BA_right_z*cos(theta_By)*sin(theta_Bz) + 2*m_arm*r_AG_right_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*r_AR1_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 2*m_arm*r_AG_right_y*cos(theta_By)*sin(phi)*sin(theta_Bz) + 4*m_rotor*r_AR1_y*cos(theta_By)*sin(phi)*sin(theta_Bz) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz))*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x)],
[                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              0,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               0,                                                                                                                                                                                                                                                                                                                                                                                             2*m_arm + m_base + 4*m_rotor,                                                                                                                                                                                                                                                                                                                                                    -2*cos(theta_By)*sin(theta_Bx)*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)),                                                                                                                                                                                                                                                                                        - 2*m_arm*r_BA_right_z*sin(theta_By) - 4*m_rotor*r_BA_right_z*sin(theta_By) - 2*m_arm*r_AG_right_y*sin(phi)*sin(theta_By) - 4*m_rotor*r_AR1_y*sin(phi)*sin(theta_By) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*cos(theta_By) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*cos(theta_By) - 2*m_arm*r_AG_right_z*cos(phi)*sin(theta_By) - 4*m_rotor*r_AR1_z*cos(phi)*sin(theta_By),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 2*cos(theta_By)*sin(theta_Bx)*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x)],
[                                                                                                                                                                                                                                                                                                                                                                                                                                                                2*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By))*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)),                                                                                                                                                                                                                                                                                                                                                                                                                                                                -2*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz))*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)),                                                                                                                                                                                                                       -2*cos(theta_By)*sin(theta_Bx)*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)), 2*I_arm_xx + I_base_xx + 4*I_rotor_xx + 2*m_arm*r_BA_right_y**2 + 2*m_arm*r_BA_right_z**2 + 4*m_rotor*r_AR1_y**2 + 4*m_rotor*r_AR1_z**2 + 4*m_rotor*r_BA_right_y**2 + 4*m_rotor*r_BA_right_z**2 + 4*m_arm*r_AG_right_y*r_BA_right_y*cos(phi) + 4*m_arm*r_AG_right_z*r_BA_right_z*cos(phi) + 8*m_rotor*r_AR1_y*r_BA_right_y*cos(phi) + 8*m_rotor*r_AR1_z*r_BA_right_z*cos(phi) + 4*m_arm*r_AG_right_y*r_BA_right_z*sin(phi) - 4*m_arm*r_AG_right_z*r_BA_right_y*sin(phi) + 8*m_rotor*r_AR1_y*r_BA_right_z*sin(phi) - 8*m_rotor*r_AR1_z*r_BA_right_y*sin(phi),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           I_base_xy + 2*I_arm_xy*cos(phi) + 2*I_arm_xz*sin(phi),                                                                                                                                                                                                                                                                                                                                            I_base_xz - 2*m_arm*r_AG_right_x*r_BA_right_z - 2*m_arm*r_BA_right_x*r_BA_right_z - 4*m_rotor*r_BA_right_x*r_BA_right_z - 2*m_arm*r_AG_right_z*r_BA_right_x*cos(phi) - 4*m_rotor*r_AR1_z*r_BA_right_x*cos(phi) - 2*m_arm*r_AG_right_y*r_BA_right_x*sin(phi) - 4*m_rotor*r_AR1_y*r_BA_right_x*sin(phi)],
[2*m_arm*r_BA_right_z*cos(theta_By)*cos(theta_Bz) - 2*m_arm*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*r_AG_right_x*sin(theta_Bx)*sin(theta_Bz) + 4*m_rotor*r_BA_right_z*cos(theta_By)*cos(theta_Bz) + 2*m_arm*r_AG_right_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 4*m_rotor*r_AR1_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 2*m_arm*r_AG_right_y*cos(theta_By)*cos(theta_Bz)*sin(phi) + 4*m_rotor*r_AR1_y*cos(theta_By)*cos(theta_Bz)*sin(phi) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By), 2*m_arm*r_AG_right_x*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*r_BA_right_z*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 4*m_rotor*r_BA_right_z*cos(theta_By)*sin(theta_Bz) + 2*m_arm*r_AG_right_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*r_AR1_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 2*m_arm*r_AG_right_y*cos(theta_By)*sin(phi)*sin(theta_Bz) + 4*m_rotor*r_AR1_y*cos(theta_By)*sin(phi)*sin(theta_Bz) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz), - 2*m_arm*r_BA_right_z*sin(theta_By) - 4*m_rotor*r_BA_right_z*sin(theta_By) - 2*m_arm*r_AG_right_y*sin(phi)*sin(theta_By) - 4*m_rotor*r_AR1_y*sin(phi)*sin(theta_By) - 2*m_arm*r_AG_right_x*cos(theta_Bx)*cos(theta_By) - 2*m_arm*r_BA_right_x*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*r_BA_right_x*cos(theta_Bx)*cos(theta_By) - 2*m_arm*r_AG_right_z*cos(phi)*sin(theta_By) - 4*m_rotor*r_AR1_z*cos(phi)*sin(theta_By),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 I_base_xy + 2*I_arm_xy*cos(phi) + 2*I_arm_xz*sin(phi),                                                                2*I_arm_zz + I_base_yy + 4*I_rotor_zz + 2*m_arm*r_BA_right_x**2 + 2*m_arm*r_BA_right_z**2 + 4*m_rotor*r_AR1_x**2 + 4*m_rotor*r_AR1_y**2 + 4*m_rotor*r_BA_right_x**2 + 4*m_rotor*r_BA_right_z**2 + 2*I_arm_yy*cos(phi)**2 - 2*I_arm_zz*cos(phi)**2 + 4*I_rotor_yy*cos(phi)**2 - 4*I_rotor_zz*cos(phi)**2 + 2*I_arm_yz*sin(2*phi) + 4*m_arm*r_AG_right_x*r_BA_right_x - 4*m_rotor*r_AR1_y**2*cos(phi)**2 + 4*m_rotor*r_AR1_z**2*cos(phi)**2 + 4*m_arm*r_AG_right_z*r_BA_right_z*cos(phi) + 8*m_rotor*r_AR1_z*r_BA_right_z*cos(phi) + 4*m_arm*r_AG_right_y*r_BA_right_z*sin(phi) + 8*m_rotor*r_AR1_y*r_BA_right_z*sin(phi) + 4*m_rotor*r_AR1_y*r_AR1_z*sin(2*phi),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        I_base_yz],
[                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                -2*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By))*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  2*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz))*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x),                                                                                                                                                                                                                                                                                                                         2*cos(theta_By)*sin(theta_Bx)*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x),                                                                                                                                                                                                                                                 I_base_xz - 2*m_arm*r_AG_right_x*r_BA_right_z - 2*m_arm*r_BA_right_x*r_BA_right_z - 4*m_rotor*r_BA_right_x*r_BA_right_z - 2*m_arm*r_AG_right_z*r_BA_right_x*cos(phi) - 4*m_rotor*r_AR1_z*r_BA_right_x*cos(phi) - 2*m_arm*r_AG_right_y*r_BA_right_x*sin(phi) - 4*m_rotor*r_AR1_y*r_BA_right_x*sin(phi),                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       I_base_yz, 2*I_arm_yy + I_base_zz + 4*I_rotor_yy + 2*m_arm*r_BA_right_x**2 + 2*m_arm*r_BA_right_y**2 + 4*m_rotor*r_AR1_x**2 + 4*m_rotor*r_AR1_z**2 + 4*m_rotor*r_BA_right_x**2 + 4*m_rotor*r_BA_right_y**2 - 2*I_arm_yy*cos(phi)**2 + 2*I_arm_zz*cos(phi)**2 - 4*I_rotor_yy*cos(phi)**2 + 4*I_rotor_zz*cos(phi)**2 - 2*I_arm_yz*sin(2*phi) + 4*m_arm*r_AG_right_x*r_BA_right_x + 4*m_rotor*r_AR1_y**2*cos(phi)**2 - 4*m_rotor*r_AR1_z**2*cos(phi)**2 + 4*m_arm*r_AG_right_y*r_BA_right_y*cos(phi) + 8*m_rotor*r_AR1_y*r_BA_right_y*cos(phi) - 4*m_arm*r_AG_right_z*r_BA_right_y*sin(phi) - 8*m_rotor*r_AR1_z*r_BA_right_y*sin(phi) - 4*m_rotor*r_AR1_y*r_AR1_z*sin(2*phi)]        
        ])
    )
    return Function("M_func",[x,phi],[out])

def b_func():
    x   = SX.sym('x',12)
    phi = SX.sym('phi',1)
    # get state variables
    x_B      = x[0]
    y_B      = x[1]
    z_B      = x[2]
    theta_Bz = x[3]
    theta_By = x[4]
    theta_Bx = x[5]
    dx_B     = x[6]
    dy_B     = x[7]
    dz_B     = x[8]
    omega_Bx = x[9]
    omega_By = x[10]
    omega_Bz = x[11]

    out = SX(
        np.array([
2*m_arm*omega_Bx*omega_Bz*r_BA_right_z*cos(theta_By)*cos(theta_Bz) - 2*m_arm*omega_Bz**2*r_AG_right_x*cos(theta_By)*cos(theta_Bz) - 2*m_arm*omega_By**2*r_BA_right_x*cos(theta_By)*cos(theta_Bz) - 2*m_arm*omega_Bz**2*r_BA_right_x*cos(theta_By)*cos(theta_Bz) - 4*m_rotor*omega_By**2*r_BA_right_x*cos(theta_By)*cos(theta_Bz) - 4*m_rotor*omega_Bz**2*r_BA_right_x*cos(theta_By)*cos(theta_Bz) - 2*m_arm*omega_Bx**2*r_BA_right_z*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_BA_right_z*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_BA_right_z*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_BA_right_z*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*omega_By**2*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*omega_By**2*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*omega_Bx**2*r_AG_right_z*cos(phi)*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_z*cos(phi)*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_AR1_z*cos(phi)*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_AR1_z*cos(phi)*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_Bx**2*r_AG_right_y*sin(phi)*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_y*sin(phi)*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_AR1_y*sin(phi)*sin(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_AR1_y*sin(phi)*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_x*cos(theta_By)*cos(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_z*cos(theta_By)*cos(theta_Bz) - 2*m_arm*omega_Bx*omega_By*r_AG_right_x*cos(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bx)*sin(theta_Bz) - 4*m_rotor*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bx)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_x*sin(theta_Bx)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x*sin(theta_Bx)*sin(theta_Bz) - 2*m_arm*omega_Bx**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*omega_By**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*omega_Bx**2*r_AR1_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 4*m_rotor*omega_By**2*r_AR1_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*omega_Bx**2*r_AG_right_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi)*sin(theta_By) - 2*m_arm*omega_By**2*r_AG_right_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi)*sin(theta_By) - 4*m_rotor*omega_Bx**2*r_AR1_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi)*sin(theta_By) - 4*m_rotor*omega_By**2*r_AR1_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi)*sin(theta_By) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_z*cos(phi)*cos(theta_By)*cos(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_y*cos(theta_By)*cos(theta_Bz)*sin(phi) - 2*m_arm*omega_By*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_Bx)*sin(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y*cos(theta_By)*cos(theta_Bz)*sin(phi) - 4*m_rotor*omega_By*omega_Bz*r_AR1_z*cos(phi)*cos(theta_Bx)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz)*sin(theta_By) - 2*m_arm*omega_By*omega_Bz*r_AG_right_y*cos(theta_Bx)*sin(phi)*sin(theta_Bz) - 4*m_rotor*omega_By*omega_Bz*r_AR1_y*cos(theta_Bx)*sin(phi)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_By*r_AG_right_x*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 2*m_arm*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 2*m_arm*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 4*m_rotor*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 2*m_arm*omega_By*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 4*m_rotor*omega_By*omega_Bz*r_AR1_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx)*sin(theta_By) + 2*m_arm*omega_By*omega_Bz*r_AG_right_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx)*sin(theta_By) + 4*m_rotor*omega_By*omega_Bz*r_AR1_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx)*sin(theta_By),
2*m_arm*omega_Bx**2*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx) - 2*m_arm*omega_Bz**2*r_AG_right_x*cos(theta_By)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_x*cos(theta_By)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_BA_right_x*cos(theta_By)*sin(theta_Bz) + 2*m_arm*omega_By**2*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx) - 2*m_arm*omega_Bz**2*r_BA_right_x*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_Bx**2*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx) - 4*m_rotor*omega_By**2*r_BA_right_x*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_By**2*r_BA_right_z*cos(theta_Bz)*sin(theta_Bx) - 4*m_rotor*omega_Bz**2*r_BA_right_x*cos(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx**2*r_AG_right_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*omega_By**2*r_AG_right_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx) + 4*m_rotor*omega_Bx**2*r_AR1_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx) + 4*m_rotor*omega_By**2*r_AR1_z*cos(phi)*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*omega_Bx**2*r_AG_right_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx) + 2*m_arm*omega_By**2*r_AG_right_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx) + 4*m_rotor*omega_Bx**2*r_AR1_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx) + 4*m_rotor*omega_By**2*r_AR1_y*cos(theta_Bz)*sin(phi)*sin(theta_Bx) - 2*m_arm*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_BA_right_z*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_BA_right_z*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_By*r_AG_right_x*cos(theta_Bx)*cos(theta_Bz) + 2*m_arm*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz) + 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*cos(theta_Bx)*cos(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_BA_right_z*cos(theta_Bx)*cos(theta_Bz) - 2*m_arm*omega_Bx*omega_Bz*r_AG_right_x*cos(theta_Bz)*sin(theta_Bx) - 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_z*cos(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bz)*sin(theta_Bx) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_z*cos(theta_By)*sin(theta_Bz) - 2*m_arm*omega_Bx**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_AR1_z*cos(phi)*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_AR1_z*cos(phi)*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*omega_Bx**2*r_AG_right_y*cos(theta_Bx)*sin(phi)*sin(theta_By)*sin(theta_Bz) - 2*m_arm*omega_By**2*r_AG_right_y*cos(theta_Bx)*sin(phi)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_Bx**2*r_AR1_y*cos(theta_Bx)*sin(phi)*sin(theta_By)*sin(theta_Bz) - 4*m_rotor*omega_By**2*r_AR1_y*cos(theta_Bx)*sin(phi)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_AR1_z*cos(phi)*cos(theta_Bx)*cos(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_AG_right_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_AR1_y*cos(theta_Bx)*cos(theta_Bz)*sin(phi) + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_z*cos(phi)*cos(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_y*cos(theta_By)*sin(phi)*sin(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y*cos(theta_By)*sin(phi)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_By*r_AG_right_x*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_Bx*omega_By*r_BA_right_x*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_BA_right_z*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_BA_right_z*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_AG_right_z*cos(phi)*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_AR1_z*cos(phi)*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 2*m_arm*omega_By*omega_Bz*r_AG_right_y*sin(phi)*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz) + 4*m_rotor*omega_By*omega_Bz*r_AR1_y*sin(phi)*sin(theta_Bx)*sin(theta_By)*sin(theta_Bz),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                2*m_arm*omega_By**2*r_AG_right_x*sin(theta_By) + 2*m_arm*omega_Bz**2*r_AG_right_x*sin(theta_By) + 2*m_arm*omega_By**2*r_BA_right_x*sin(theta_By) + 2*m_arm*omega_Bz**2*r_BA_right_x*sin(theta_By) + 4*m_rotor*omega_By**2*r_BA_right_x*sin(theta_By) + 4*m_rotor*omega_Bz**2*r_BA_right_x*sin(theta_By) - 2*m_arm*omega_Bx*omega_Bz*r_BA_right_z*sin(theta_By) - 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_z*sin(theta_By) - 2*m_arm*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*cos(theta_By) - 2*m_arm*omega_By**2*r_BA_right_z*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*omega_Bx**2*r_BA_right_z*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*omega_By**2*r_BA_right_z*cos(theta_Bx)*cos(theta_By) - 2*m_arm*omega_Bx**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*cos(theta_By) - 2*m_arm*omega_By**2*r_AG_right_z*cos(phi)*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*omega_Bx**2*r_AR1_z*cos(phi)*cos(theta_Bx)*cos(theta_By) - 4*m_rotor*omega_By**2*r_AR1_z*cos(phi)*cos(theta_Bx)*cos(theta_By) - 2*m_arm*omega_Bx**2*r_AG_right_y*cos(theta_Bx)*cos(theta_By)*sin(phi) - 2*m_arm*omega_By**2*r_AG_right_y*cos(theta_Bx)*cos(theta_By)*sin(phi) - 4*m_rotor*omega_Bx**2*r_AR1_y*cos(theta_Bx)*cos(theta_By)*sin(phi) - 4*m_rotor*omega_By**2*r_AR1_y*cos(theta_Bx)*cos(theta_By)*sin(phi) + 2*m_arm*omega_Bx*omega_Bz*r_AG_right_x*cos(theta_Bx)*cos(theta_By) + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*cos(theta_By) + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x*cos(theta_Bx)*cos(theta_By) - 2*m_arm*omega_Bx*omega_Bz*r_AG_right_z*cos(phi)*sin(theta_By) - 4*m_rotor*omega_Bx*omega_Bz*r_AR1_z*cos(phi)*sin(theta_By) + 2*m_arm*omega_Bx*omega_By*r_AG_right_x*cos(theta_By)*sin(theta_Bx) + 2*m_arm*omega_Bx*omega_By*r_BA_right_x*cos(theta_By)*sin(theta_Bx) + 2*m_arm*omega_By*omega_Bz*r_BA_right_z*cos(theta_By)*sin(theta_Bx) + 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*cos(theta_By)*sin(theta_Bx) + 4*m_rotor*omega_By*omega_Bz*r_BA_right_z*cos(theta_By)*sin(theta_Bx) - 2*m_arm*omega_Bx*omega_Bz*r_AG_right_y*sin(phi)*sin(theta_By) - 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y*sin(phi)*sin(theta_By) + 2*m_arm*omega_By*omega_Bz*r_AG_right_z*cos(phi)*cos(theta_By)*sin(theta_Bx) + 4*m_rotor*omega_By*omega_Bz*r_AR1_z*cos(phi)*cos(theta_By)*sin(theta_Bx) + 2*m_arm*omega_By*omega_Bz*r_AG_right_y*cos(theta_By)*sin(phi)*sin(theta_Bx) + 4*m_rotor*omega_By*omega_Bz*r_AR1_y*cos(theta_By)*sin(phi)*sin(theta_Bx),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     I_base_yz*omega_By**2 - I_base_yz*omega_Bz**2 - I_base_xy*omega_Bx*omega_Bz + I_base_xz*omega_Bx*omega_By - I_base_yy*omega_By*omega_Bz + I_base_zz*omega_By*omega_Bz - 2*I_arm_xy*omega_Bx*omega_Bz*cos(phi) - 2*I_arm_xz*omega_Bx*omega_Bz*sin(phi) + 2*m_arm*omega_By*omega_Bz*r_BA_right_y**2 - 2*m_arm*omega_By*omega_Bz*r_BA_right_z**2 + 4*m_rotor*omega_By*omega_Bz*r_BA_right_y**2 - 4*m_rotor*omega_By*omega_Bz*r_BA_right_z**2 - 2*I_arm_yy*omega_By*omega_Bz*cos(2*phi) + 2*I_arm_zz*omega_By*omega_Bz*cos(2*phi) - 4*I_rotor_yy*omega_By*omega_Bz*cos(2*phi) + 4*I_rotor_zz*omega_By*omega_Bz*cos(2*phi) - 4*I_arm_yz*omega_By*omega_Bz*sin(2*phi) - 2*m_arm*omega_Bx*omega_By*r_AG_right_x*r_BA_right_z - 2*m_arm*omega_Bx*omega_By*r_BA_right_x*r_BA_right_z - 4*m_rotor*omega_Bx*omega_By*r_BA_right_x*r_BA_right_z + 4*m_rotor*omega_By*omega_Bz*r_AR1_y**2*cos(2*phi) - 4*m_rotor*omega_By*omega_Bz*r_AR1_z**2*cos(2*phi) - 2*m_arm*omega_Bx*omega_By*r_AG_right_z*r_BA_right_x*cos(phi) + 4*m_arm*omega_By*omega_Bz*r_AG_right_y*r_BA_right_y*cos(phi) - 4*m_arm*omega_By*omega_Bz*r_AG_right_z*r_BA_right_z*cos(phi) - 4*m_rotor*omega_Bx*omega_By*r_AR1_z*r_BA_right_x*cos(phi) + 8*m_rotor*omega_By*omega_Bz*r_AR1_y*r_BA_right_y*cos(phi) - 8*m_rotor*omega_By*omega_Bz*r_AR1_z*r_BA_right_z*cos(phi) - 2*m_arm*omega_Bx*omega_By*r_AG_right_y*r_BA_right_x*sin(phi) - 4*m_arm*omega_By*omega_Bz*r_AG_right_y*r_BA_right_z*sin(phi) - 4*m_arm*omega_By*omega_Bz*r_AG_right_z*r_BA_right_y*sin(phi) - 4*m_rotor*omega_Bx*omega_By*r_AR1_y*r_BA_right_x*sin(phi) - 8*m_rotor*omega_By*omega_Bz*r_AR1_y*r_BA_right_z*sin(phi) - 8*m_rotor*omega_By*omega_Bz*r_AR1_z*r_BA_right_y*sin(phi) - 8*m_rotor*omega_By*omega_Bz*r_AR1_y*r_AR1_z*sin(2*phi),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             I_base_xz*omega_Bz**2 - I_base_xz*omega_Bx**2 + 2*I_arm_xx*omega_Bx*omega_Bz - 2*I_arm_yy*omega_Bx*omega_Bz + I_base_xx*omega_Bx*omega_Bz + I_base_xy*omega_By*omega_Bz - I_base_yz*omega_Bx*omega_By - I_base_zz*omega_Bx*omega_Bz + 4*I_rotor_xx*omega_Bx*omega_Bz - 4*I_rotor_yy*omega_Bx*omega_Bz + 2*I_arm_xy*omega_By*omega_Bz*cos(phi) + 2*I_arm_xz*omega_By*omega_Bz*sin(phi) - 2*m_arm*omega_Bx*omega_Bz*r_BA_right_x**2 + 2*m_arm*omega_Bx*omega_Bz*r_BA_right_z**2 - 4*m_rotor*omega_Bx*omega_Bz*r_AR1_x**2 + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y**2 - 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_x**2 + 4*m_rotor*omega_Bx*omega_Bz*r_BA_right_z**2 + 2*m_arm*omega_Bx**2*r_AG_right_x*r_BA_right_z - 2*m_arm*omega_Bz**2*r_AG_right_x*r_BA_right_z + 2*m_arm*omega_Bx**2*r_BA_right_x*r_BA_right_z - 2*m_arm*omega_Bz**2*r_BA_right_x*r_BA_right_z + 4*m_rotor*omega_Bx**2*r_BA_right_x*r_BA_right_z - 4*m_rotor*omega_Bz**2*r_BA_right_x*r_BA_right_z + 2*I_arm_yy*omega_Bx*omega_Bz*cos(phi)**2 - 2*I_arm_zz*omega_Bx*omega_Bz*cos(phi)**2 + 4*I_rotor_yy*omega_Bx*omega_Bz*cos(phi)**2 - 4*I_rotor_zz*omega_Bx*omega_Bz*cos(phi)**2 + 2*I_arm_yz*omega_Bx*omega_Bz*sin(2*phi) + 2*m_arm*omega_Bx**2*r_AG_right_z*r_BA_right_x*cos(phi) - 2*m_arm*omega_Bz**2*r_AG_right_z*r_BA_right_x*cos(phi) + 4*m_rotor*omega_Bx**2*r_AR1_z*r_BA_right_x*cos(phi) - 4*m_rotor*omega_Bz**2*r_AR1_z*r_BA_right_x*cos(phi) + 2*m_arm*omega_Bx**2*r_AG_right_y*r_BA_right_x*sin(phi) - 2*m_arm*omega_Bz**2*r_AG_right_y*r_BA_right_x*sin(phi) + 4*m_rotor*omega_Bx**2*r_AR1_y*r_BA_right_x*sin(phi) - 4*m_rotor*omega_Bz**2*r_AR1_y*r_BA_right_x*sin(phi) - 4*m_arm*omega_Bx*omega_Bz*r_AG_right_x*r_BA_right_x - 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y**2*cos(phi)**2 + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_z**2*cos(phi)**2 + 4*m_arm*omega_Bx*omega_Bz*r_AG_right_z*r_BA_right_z*cos(phi) + 8*m_rotor*omega_Bx*omega_Bz*r_AR1_z*r_BA_right_z*cos(phi) + 4*m_arm*omega_Bx*omega_Bz*r_AG_right_y*r_BA_right_z*sin(phi) + 8*m_rotor*omega_Bx*omega_Bz*r_AR1_y*r_BA_right_z*sin(phi) + 4*m_rotor*omega_Bx*omega_Bz*r_AR1_y*r_AR1_z*sin(2*phi),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 I_base_xy*omega_Bx**2 - I_base_xy*omega_By**2 + 2*I_arm_xy*omega_Bx**2*cos(phi) - 2*I_arm_xy*omega_By**2*cos(phi) + 2*I_arm_xz*omega_Bx**2*sin(phi) - 2*I_arm_xz*omega_By**2*sin(phi) - 2*I_arm_xx*omega_Bx*omega_By + 2*I_arm_zz*omega_Bx*omega_By - I_base_xx*omega_Bx*omega_By + I_base_yy*omega_Bx*omega_By - I_base_xz*omega_By*omega_Bz + I_base_yz*omega_Bx*omega_Bz - 4*I_rotor_xx*omega_Bx*omega_By + 4*I_rotor_zz*omega_Bx*omega_By + 2*m_arm*omega_Bx*omega_By*r_BA_right_x**2 - 2*m_arm*omega_Bx*omega_By*r_BA_right_y**2 + 4*m_rotor*omega_Bx*omega_By*r_AR1_x**2 - 4*m_rotor*omega_Bx*omega_By*r_AR1_z**2 + 4*m_rotor*omega_Bx*omega_By*r_BA_right_x**2 - 4*m_rotor*omega_Bx*omega_By*r_BA_right_y**2 + 2*I_arm_yy*omega_Bx*omega_By*cos(phi)**2 - 2*I_arm_zz*omega_Bx*omega_By*cos(phi)**2 + 4*I_rotor_yy*omega_Bx*omega_By*cos(phi)**2 - 4*I_rotor_zz*omega_Bx*omega_By*cos(phi)**2 + 2*I_arm_yz*omega_Bx*omega_By*sin(2*phi) + 4*m_arm*omega_Bx*omega_By*r_AG_right_x*r_BA_right_x + 2*m_arm*omega_By*omega_Bz*r_AG_right_x*r_BA_right_z + 2*m_arm*omega_By*omega_Bz*r_BA_right_x*r_BA_right_z + 4*m_rotor*omega_By*omega_Bz*r_BA_right_x*r_BA_right_z - 4*m_rotor*omega_Bx*omega_By*r_AR1_y**2*cos(phi)**2 + 4*m_rotor*omega_Bx*omega_By*r_AR1_z**2*cos(phi)**2 - 4*m_arm*omega_Bx*omega_By*r_AG_right_y*r_BA_right_y*cos(phi) + 2*m_arm*omega_By*omega_Bz*r_AG_right_z*r_BA_right_x*cos(phi) - 8*m_rotor*omega_Bx*omega_By*r_AR1_y*r_BA_right_y*cos(phi) + 4*m_rotor*omega_By*omega_Bz*r_AR1_z*r_BA_right_x*cos(phi) + 4*m_arm*omega_Bx*omega_By*r_AG_right_z*r_BA_right_y*sin(phi) + 2*m_arm*omega_By*omega_Bz*r_AG_right_y*r_BA_right_x*sin(phi) + 8*m_rotor*omega_Bx*omega_By*r_AR1_z*r_BA_right_y*sin(phi) + 4*m_rotor*omega_By*omega_Bz*r_AR1_y*r_BA_right_x*sin(phi) + 4*m_rotor*omega_Bx*omega_By*r_AR1_y*r_AR1_z*sin(2*phi)        
        ])
    )
    return Function("b_func",[x,phi],[out])

def g_func():
    x   = SX.sym('x',12)
    phi = SX.sym('phi',1)
    # get state variables
    x_B      = x[0]
    y_B      = x[1]
    z_B      = x[2]
    theta_Bz = x[3]
    theta_By = x[4]
    theta_Bx = x[5]
    dx_B     = x[6]
    dy_B     = x[7]
    dz_B     = x[8]
    omega_Bx = x[9]
    omega_By = x[10]
    omega_Bz = x[11]

    out = SX(
        np.array([
                                                                                                                                                                                                                                                                                                                                                                                                                       0,
                                                                                                                                                                                                                                                                                                                                                                                                                       0,
                                                                                                                                                                                                                                                                                                                                                                                 -gravity*(2*m_arm + m_base + 4*m_rotor),
                                                                                                                                                                                                               2*gravity*cos(theta_By)*sin(theta_Bx)*(m_arm*r_BA_right_z + 2*m_rotor*r_BA_right_z + m_arm*r_AG_right_z*cos(phi) + 2*m_rotor*r_AR1_z*cos(phi) + m_arm*r_AG_right_y*sin(phi) + 2*m_rotor*r_AR1_y*sin(phi)),
2*gravity*(m_arm*r_BA_right_z*sin(theta_By) + 2*m_rotor*r_BA_right_z*sin(theta_By) + m_arm*r_AG_right_y*sin(phi)*sin(theta_By) + 2*m_rotor*r_AR1_y*sin(phi)*sin(theta_By) + m_arm*r_AG_right_x*cos(theta_Bx)*cos(theta_By) + m_arm*r_BA_right_x*cos(theta_Bx)*cos(theta_By) + 2*m_rotor*r_BA_right_x*cos(theta_Bx)*cos(theta_By) + m_arm*r_AG_right_z*cos(phi)*sin(theta_By) + 2*m_rotor*r_AR1_z*cos(phi)*sin(theta_By)),
                                                                                                                                                                                                                                                                                                               -2*gravity*cos(theta_By)*sin(theta_Bx)*(m_arm*r_AG_right_x + m_arm*r_BA_right_x + 2*m_rotor*r_BA_right_x)      
        ])
    )
    return Function("g",[x,phi],[out])

def S_func():
    x   = SX.sym('x',12)
    phi = SX.sym('phi',1)
    # get state variables
    x_B      = x[0]
    y_B      = x[1]
    z_B      = x[2]
    theta_Bz = x[3]
    theta_By = x[4]
    theta_Bx = x[5]
    dx_B     = x[6]
    dy_B     = x[7]
    dz_B     = x[8]
    omega_Bx = x[9]
    omega_By = x[10]
    omega_Bz = x[11]

    out = SX(
        np.array([
[- kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)) - kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) + kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi + theta_Bx)*cos(theta_By), -kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_AR1_x*cos(phi) + r_BA_right_x*cos(phi) - kM*sin(phi)),  kT*(kM*cos(phi) + r_AR1_x*sin(phi) + r_BA_right_x*sin(phi))],
[  kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)) - kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) - kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi - theta_Bx)*cos(theta_By),  kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_BA_right_x*cos(phi) - r_AR1_x*cos(phi) + kM*sin(phi)),  kT*(kM*cos(phi) + r_AR1_x*sin(phi) - r_BA_right_x*sin(phi))],
[  kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)) - kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) - kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi - theta_Bx)*cos(theta_By),  kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_AR1_x*cos(phi) + r_BA_right_x*cos(phi) - kM*sin(phi)), -kT*(kM*cos(phi) + r_AR1_x*sin(phi) + r_BA_right_x*sin(phi))],
[- kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)) - kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) + kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi + theta_Bx)*cos(theta_By), -kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_BA_right_x*cos(phi) - r_AR1_x*cos(phi) + kM*sin(phi)), -kT*(kM*cos(phi) + r_AR1_x*sin(phi) - r_BA_right_x*sin(phi))]        
        ])
    )
    return Function("S_func",[x,phi],[out])

def F_func():
    theta_Bx = SX.sym('theta_Bx',1)
    theta_By = SX.sym('theta_By',1)
    theta_Bz = SX.sym('theta_Bz',1)
    # transforms omega to chi
    out = SX(
        np.array([
        [0,                 sin(theta_Bx)/cos(theta_By),                 cos(theta_Bx)/cos(theta_By)],
        [0,                               cos(theta_Bx),                              -sin(theta_Bx)],
        [1, (sin(theta_Bx)*sin(theta_By))/cos(theta_By), (cos(theta_Bx)*sin(theta_By))/cos(theta_By)]
    ])
    )
    return Function("F_func",[theta_Bx,theta_By,theta_Bz],[out])

def G_func():
    theta_Bx = SX.sym('theta_Bx',1)
    theta_By = SX.sym('theta_By',1)
    theta_Bz = SX.sym('theta_Bz',1)
    # transforms chi to omega
    out = SX(
        np.array([
            [     -sin(theta_By),              0, 1],
            [cos(theta_By)*sin(theta_Bx),  cos(theta_Bx), 0],
            [cos(theta_Bx)*cos(theta_By), -sin(theta_Bx), 0]
        ])
    )
    return Function("G_func",[theta_Bx,theta_By,theta_Bz],[out])

def dynamics_func():
    # get symbolic variables
    X      = MX.sym('X',12,1)
    U      = MX.sym('U',4,1)
    varphi = MX.sym('varphi',1,1)

    # get state variables
    x_B      = X[0]
    y_B      = X[1]
    z_B      = X[2]
    theta_Bz = X[3]
    theta_By = X[4]
    theta_Bx = X[5]
    dx_B     = X[6]
    dy_B     = X[7]
    dz_B     = X[8]
    omega_Bx = X[9]
    omega_By = X[10]
    omega_Bz = X[11]

    # q, u
    q = vertcat(x_B,y_B,z_B,theta_Bz,theta_By,theta_Bx)
    u = vertcat(dx_B,dy_B,dz_B,omega_Bx,omega_By,omega_Bz)

    # get dynamics matrices
    M_x = M_func()(X,varphi)
    b_x = b_func()(X,varphi)
    g_x = g_func()(X,varphi)  
    S_x = S_func()(X,varphi)

    # compute inverse mass matrix
    Minv = inv(M_x)
    du = Minv @ (S_x.T @ U - b_x - g_x)

    chid = F_func()(theta_Bx,theta_By,theta_Bz) @ vertcat(omega_Bx,omega_By,omega_Bz)

    # finally write explicit dynamics
    f_expl = vertcat(u[0],u[1],u[2],chid,du)
    
    return Function('dynamics_func',[X,U,varphi],[f_expl])

def export_robot_model() -> AcadosModel:
    model_name = "atmo"

    # tilt angle
    varphi = MX.sym("varphi",1)
    p = varphi

    # states and controls
    X = MX.sym("X",12)
    U = MX.sym("U",4)

    # Explicit nominal dynamics
    f_expl = dynamics_func()(X,U,varphi)

    # xdot
    x_B_dot      = MX.sym("x_B_dot")
    y_B_dot      = MX.sym("y_B_dot")
    z_B_dot      = MX.sym("z_B_dot")
    theta_Bz_dot = MX.sym("theta_Bz_dot")
    theta_By_dot = MX.sym("theta_By_dot")
    theta_Bx_dot = MX.sym("theta_Bx_dot")
    dx_B_dot     = MX.sym("dx_B_dot")
    dy_B_dot     = MX.sym("dy_B_dot")
    dz_B_dot     = MX.sym("dz_B_dot")
    omega_Bx_dot = MX.sym("omega_Bx_dot")
    omega_By_dot = MX.sym("omega_By_dot")
    omega_Bz_dot = MX.sym("omega_Bz_dot")

    Xdot = vertcat(x_B_dot,y_B_dot,z_B_dot,theta_Bz_dot,theta_By_dot,theta_Bx_dot,dx_B_dot,dy_B_dot,dz_B_dot,omega_Bx_dot,omega_By_dot,omega_Bz_dot)

    # write implicit dynamics 
    f_impl = Xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = X
    model.xdot = Xdot
    model.u = U
    model.p = p
    model.name = model_name

    return model

def create_ocp_solver_description() -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # model = export_robot_model()
    model = export_robot_model()

    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat_terminal
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.hstack((x_ref,u_ref))
    ocp.cost.yref_e = x_ref

    # set constraints
    ocp.constraints.lbu = np.array([0,0,0,0])
    ocp.constraints.ubu = np.array([+u_max,+u_max,+u_max,+u_max])
    ocp.constraints.idxbu = np.array([0,1,2,3])
    
    ocp.constraints.x0 = x0

    # parameters
    ocp.parameter_values = np.array([varphi0])
    
    # set options
    ocp.solver_options.qp_solver       = "FULL_CONDENSING_HPIPM"       # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx  = "GAUSS_NEWTON"                # GAUSS_NEWTON, EXACT
    ocp.solver_options.integrator_type = "ERK"                         # ERK      
    ocp.solver_options.nlp_solver_type = "SQP_RTI"                     # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = T_horizon

    # export directory
    ocp.code_export_directory = acados_ocp_path

    return ocp