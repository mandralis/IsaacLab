import scipy
import numpy as np
from numpy.linalg import pinv
from casadi import SX, vertcat, sin, cos, inv, Function, MX
from atmo_parameters import params_
from IPython import embed
from scipy.linalg import null_space


kT                   = params_.get('kT')
kM                   = params_.get('kM')
r_BA_right_x         = params_.get('r_BA_right_x')
r_BA_right_y         = params_.get('r_BA_right_y')
r_BA_right_z         = params_.get('r_BA_right_z')
r_AG_right_x         = params_.get('r_AG_right_x')
r_AG_right_y         = params_.get('r_AG_right_y')
r_AG_right_z         = params_.get('r_AG_right_z')
r_AR1_x              = params_.get('r_AR1_x')
r_AR1_y              = params_.get('r_AR1_y')
r_AR1_z              = params_.get('r_AR1_z')

def S_func():
    x   = SX.sym('x',3)
    phi = SX.sym('phi',1)
    theta_Bz = x[0]
    theta_By = x[1]
    theta_Bx = x[2]

    out = SX(
        np.array([
[- kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)) - kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) + kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi + theta_Bx)*cos(theta_By), -kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_AR1_x*cos(phi) + r_BA_right_x*cos(phi) - kM*sin(phi)),  kT*(kM*cos(phi) + r_AR1_x*sin(phi) + r_BA_right_x*sin(phi))],
[  kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)) - kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) - kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi - theta_Bx)*cos(theta_By),  kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_BA_right_x*cos(phi) - r_AR1_x*cos(phi) + kM*sin(phi)),  kT*(kM*cos(phi) + r_AR1_x*sin(phi) - r_BA_right_x*sin(phi))],
[  kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)) - kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) - kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi - theta_Bx)*cos(theta_By),  kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_AR1_x*cos(phi) + r_BA_right_x*cos(phi) - kM*sin(phi)), -kT*(kM*cos(phi) + r_AR1_x*sin(phi) + r_BA_right_x*sin(phi))],
[- kT*cos(phi)*(sin(theta_Bx)*sin(theta_Bz) + cos(theta_Bx)*cos(theta_Bz)*sin(theta_By)) - kT*sin(phi)*(cos(theta_Bx)*sin(theta_Bz) - cos(theta_Bz)*sin(theta_Bx)*sin(theta_By)), kT*cos(phi)*(cos(theta_Bz)*sin(theta_Bx) - cos(theta_Bx)*sin(theta_By)*sin(theta_Bz)) + kT*sin(phi)*(cos(theta_Bx)*cos(theta_Bz) + sin(theta_Bx)*sin(theta_By)*sin(theta_Bz)), -kT*cos(phi + theta_Bx)*cos(theta_By), -kT*(r_AR1_y + r_BA_right_y*cos(phi) + r_BA_right_z*sin(phi)), kT*(r_BA_right_x*cos(phi) - r_AR1_x*cos(phi) + kM*sin(phi)), -kT*(kM*cos(phi) + r_AR1_x*sin(phi) - r_BA_right_x*sin(phi))]        
        ])
    )
    return Function("S_func",[x,phi],[out])

# Get control allocation matrix CA 
S = S_func()

def weighted_pinv(A, W=None):
    """Compute weighted pseudo-inverse."""
    if W is None:
        W = np.eye(A.shape[0])
    return np.linalg.pinv(W @ A) @ W

class BRTC:
    def __init__(self, 
                 Kp=np.array([10.0,10.0,14.0]), 
                 Kd=np.array([0.3,0.3,0.3]), 
                 Ki=np.array([0.6,0.6,0.2]), 
                 h=0.01):
        # filter constants
        Tf = (Kd / Kp) / 10
        kaw = 0.001 * Ki
        Taw = h / kaw

        # controller coefficients
        self.Kp   = Kp
        self.bi   = Ki * h
        self.ad   = Tf / (Tf + h)
        self.bd   = Kd / (Tf + h)
        self.br   = h / Taw 
        self.b    = 1.0            # no setpoint weighting

        # controller states
        self.P     = np.zeros(3)
        self.I     = np.zeros(3)
        self.D     = np.zeros(3)
        self.omega = np.zeros(3)

    def advance(self, phi, omega, omega_des, c_des):
        # compute control allocation matrix
        CA = np.array(S(np.array([0.0,0.0,0.0]), phi)).transpose()[2:6,:]

        # perform PID step
        self.P          = self.Kp * (self.b * omega_des - omega)
        self.D          = self.ad * self.D - self.bd * (omega - self.omega)
        tau_des         = self.P + self.I + self.D
        ua              = pinv(CA) @ np.hstack([c_des,tau_des])
        u               = np.clip(ua, 0, 1)
        sat_error       = CA @ (u - ua)  
        self.I          = self.I + self.bi * (omega_des - omega) + self.br * sat_error[1:]
        self.omega      = omega
        return u
