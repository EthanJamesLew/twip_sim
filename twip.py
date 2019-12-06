'''TWIP Model

Ethan Lew
1/26/2019
elew@pdx.edu 

Naming Conventions:

Differentials: d<variable> (e.g. dt, dp)
Differential Systems: vd<coord> (e.g. vdq)  
Coordinates: q is preferred for generalized DYNAMICS coordinates, p is preferred for POSITIONAL coordinates
Current States: c<name> (e.g. ct)

Style Conventions: 

In general, prefer snake case over camel case (except for class names). Uphold the standards outlined by PEP8 when
it is logical to do so.  Math functionality should use be short and ideally one word. Public interface functions 
should have the proper accessor conventions and be descriptive in general.

Comments in implementation should be formatted as

# <quick comment>
code

Comments over classes should be formatted as

class className(object):
'' '<Verbose Description>
'' '

'''

import numpy as np
from numpy import sin, cos
from system import SysBase, wraptopi

class TWIPZi(SysBase):
    '''TWIP System described in Z. Li et al, Advanced Control of Wheeled Inverted Pendulums, Springer-Verlag London 2013
    
    The parameter schema is not fixed as different algorithms require different parameters for their outcomes. 
    For rendering to work, the following values should be defined somewhere:

        l - distance from the track to the COG
        d - length of the track
        r - radius of the wheel

    The TWIP system can have a variety of generalized coordinate descriptions. Only one interpretation can be used by the
    viewer, though, so a method is available to convert to this form (get_position_coordinates)

        (x, y) - position of the TWIP
        theta  - TWIP yaw angle
        alpha  - TWIP tilt angle
        thetar - right wheel angle
        thetal - left wheel angle
    '''
    def __init__(self):
        SysBase.__init__(self, n=6)
        default_bot = {"Mw": 0.02, "Iw": 33.6e-6, "r" : .04, "m" : 0.0,
             "l" : .047, "d" : .161, "M" : 1.22, "IM": 1.776e-3,
                "Ip": 2.601e-3, "g": 9.81 }
        self.parameters = default_bot
        self.equations = "Zi et al."
        self.q = np.zeros((6))
        self.p = np.zeros((5))
        self.kinematic_coordinates = np.zeros((5, 1))
        self.force = np.zeros((4, 1))

        self.update_params()

    def get_position_coordinates(self):
        return np.concatenate((self.p, [self.q[2]]))

    def vdp(self, t, q, ic):
        theta = self.q[1]
        v = self.q[3]
        omega = self.q[4]
        rbt = self.parameters
        r =    rbt['r']
        d =     rbt['d']
        
        qp = np.zeros((5))
        qp[0] = cos(theta)*v
        qp[1] = sin(theta)*v 
        qp[2] = omega
        qp[3] = v/r + omega/d 
        qp[4] = v/r - omega/d 
        return qp

    def update_params(self):
        # Unpack Parameters
        rbt = self.parameters
        self.M =     rbt['M']
        self.Mw =    rbt['Mw']
        self.m =     rbt['m']
        self.Iw =    rbt['Iw']
        self.Ip =    rbt['Ip']
        self.Imm =    rbt['IM']
        self.r =    rbt['r']
        self.l =     rbt['l']
        self.d =     rbt['d']
        self.g = rbt['g']

    def vdq(self, t, q, F):

        # Unpack forcing parameters
        tl =    F[0]
        tr  =    F[1]
        dl =    F[2]
        dr  =   F[3]
        
        # Calculate differential
        qp = np.zeros((6))
        qp[0] = q[3]
        qp[1] = q[4]
        qp[2] = q[5]
        qp[3] =((self.m*self.l**2+self.Imm)*(-self.m*self.l*q[5]**2*sin(q[2])-tl/self.r-tr/self.r-dl-dr)+self.m**2*self.l**2*cos(q[2])*self.g*sin(q[2]))/((self.m*self.l**2+self.Imm)*(self.M+2*self.Mw+self.m+2*self.Iw/self.r**2)-self.m**2*self.l**2*cos(q[2])**2)
        qp[4] = 2*self.d*(tl/self.r-tr/self.r+dl-dr)/(self.Ip+(2*(self.Mw+self.Iw/self.r**2))*self.d**2)
        qp[5] =  (self.m*self.l*cos(q[2])*(-self.m*self.l*q[5]**2*sin(q[2])-tl/self.r-tr/self.r-dl-dr)+self.m*self.g*self.l*sin(q[2])*(self.M+2*self.Mw+self.m+2*self.Iw/self.r**2))/((self.m*self.l**2+self.Imm)*(self.M+2*self.Mw+self.m+2*self.Iw/self.r**2)-self.m**2*self.l**2*cos(q[2])**2) 
                
        return qp

    def convert_sys(self):
        self.p[2] = self.q[1]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    twip = TWIPZi()

    # Simulation Parameters
    nsteps = 340
    dt = 1/60.
    t = np.zeros(nsteps)

    pos = np.zeros((nsteps, 2))
    twip.set_IC([0, 3.14/4, 0, 0, 0, 0])

    # Fake impulse
    twip.update_current_state(dt, [1/dt*0.1, 1/dt*0.5,  .2, 0]) 

    for i in range(0, nsteps):
        cpos = twip.get_position_coordinates()
        pos[i, 0] = cpos[0]
        pos[i, 1] = cpos[1]
        t[i] = i*dt
        twip.update_current_state(dt, [0, 0,  0, 0])

    plt.plot(t, pos[:, 1])
    plt.show()