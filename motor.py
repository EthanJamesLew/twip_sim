'''DC Motor Model

Ethan Lew
4/14/2019
elew@pdx.edu
Credit to Justin Patterson (jp32@pdx.edu) for providing this model and measuring the motor parameters.
'''

import numpy as np
from numpy import sqrt
from system import SysBase

import time
class DCMotor(SysBase):
    ''' DC Motor Description

    The DC motor description incoporates the motor parameters:

        J (kg m^2) - Rotational inertia of the load attached to the armature
        L (H) - Motor Inductance
        R (Ohms) - Motor Resistance
        Km (V s) - proportionality constant tau = Km * I
        Ke (V s / rad) - proportionality constant EMF = Ke * omega

    Note that the simulator is unitless, so the units provided are examples.
    '''
    def __init__ (self):
        SysBase.__init__(self, n=2)
        # these values were taken from a pololu motor: https://www.pololu.com/product/3202, 550 RPM model
        #default_motor = {'J' : 127e-6, 'L': 15.38e-3, 'R': 11.1, 'Km': 0.0993, 'Ke': 0.1893, 'd':0.0}

        default_motor = {'J' : 32e-6, 'L': 15.38e-3, 'R': 11, 'Km': 0.0993, 'Ke': 0.1893, 'd':0.0, 'b':0.01}
        self.parameters = default_motor
        self.equations = 'Vukosavic'

        # holds [current, angular rate]
        self.q = np.zeros((2, 1))
        # holds [torque]
        
        self.p = np.zeros((1, 1))
        self.kinematic_coordinates = np.zeros((1, 1))

        # holds [Vrms]
        self.force = np.zeros((1))

        self.update_params()


    def update_params(self):
        mtr = self.parameters
        self.R = mtr['R']
        self.L = mtr['L']
        self.Ke = mtr['Ke']
        self.Km = mtr['Km']
        self.J = mtr['J']
        self.d = mtr['d']
        self.b = mtr['b']

    def vdq(self, t, q, F):

        if (abs(F[0]) < self.d):
            F = [0]

        dq = np.zeros((2))
        dq[0] = (-self.R/self.L)*q[0] - (self.Ke/self.L)*q[1] + 1/self.L*F[0]
        dq[1] =  (self.Km/self.J)*q[0] - (self.b / self.J)*q[1]
        return dq

    def convert_sys(self):
        self.p = self.parameters['Km'] * self.q 

    def vdp(self, t, q, ic):
        return np.zeros((2))

class PWMDCMotor(DCMotor):
    ''' PWM DC Motor

    This is a very simple converter that accepts 8 bit duty cycle numbers and 
    converts it into Vrms for the DCMotor model. No new dynamics are introduced.

    Assumes a very fast frequency, as the switching is not modeled
    '''
    def __init__(self):
        DCMotor.__init__(self)
        self.parameters['resolution'] = 8
        self.parameters['Vs'] = 24

    def convert_pwm(self, F):
        Vs = self.parameters['Vs']
        res = self.parameters['resolution']


        sign = np.sign(F[0])
        pwm = min(max(abs(F[0]), 0), 2**res-1)
        #print(sign*pwm, F)
        return np.array([sign*Vs*sqrt(pwm/(2**res-1))])

    def vdq(self, t, q, F):
        return super(PWMDCMotor, self).vdq( t, q, self.convert_pwm(F))

    def set_force(self, F):
        self.force = self.convert_pwm(F)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    motor = PWMDCMotor()

    print(motor)

    nsteps = 3400
    dt = 1/10.
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 2))
    motor.set_IC([0, 0])

    start = time.time()

    for i in range(0, nsteps):
        pos[i, 0] = i*dt
        pos[i, 1] = motor.get_position_coordinates()[0]
        t[i] = i*dt
        motor.update_current_state(dt, [-255])

    end = time.time()

    print("Time to execute %2.6f" % (end - start))
        

    plt.plot(pos[:, 0], pos[:, 1])
    plt.show()