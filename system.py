from numpy import floor
from scipy.integrate import odeint
#from diffeqpy import de

def wraptopi(x):
    '''Phase wrapping method from [-pi, pi]

    Args: 
        x: a continuous phase value in R
    Returns:
        X: a wrapped phase value in [-pi, pi]
    '''
    pi = 3.141592
    x = x - floor(x/(2*pi)) *2 *pi
    if x > pi:
        x = x - 2*pi
    return x 


def rk4(f):
    '''Generic implementation of the Runge-Kutta solver with Dormand-Prince weights

    Args:
        f: A function f(t, y), such that y' = f(t, y)
    Returns:
        y: solution 

    This system doesn't have any error free guarantees that can be found in scipy's integrate.solve_ivp() method. In 
    general, scipy's solution isn't ideal for the simulation approach used here. Adaptive time methods may be worth
    exploring in the future.
    '''
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
                        )( dt * f( t + dt  , y + dy3   ) )
	                    )( dt * f( t + dt/2, y + dy2/2 ) )
	                    )( dt * f( t + dt/2, y + dy1/2 ) )
	                    )( dt * f( t       , y         ) )

'''
System Custom Error 
'''
class SysNoParameterError(Exception):
   pass

class SysBase(object):
    '''Holds the interface, parameters and initial conditions for the simulation of a system. 
    
    Variables:
        q - dynamics space
        p - kinematics space
        force - system input

    Dynamics coordinates need to be held in self.q and kinematic (positional) coordinates are held in self.p

    Implementation names (authors preferred) are held in a field called self.equations. Cite responsibly.

    Needs: SysBase describes the state of a system over time. To use properly, a timing system is needed. Also, a
    controller class should be responsible for selecting the F values to achieve stability. Finally, an optional sensor 
    class can model the delay and noise present in a realized TWIP system.
    '''
    def __init__(self, n = 3):
        default_bot = {"Mw": 0.8, "Iw": 0.02, "r" : .2, "m" : 0.5,
             "l" : .15, "d" : .6, "M" : 0.7, "IM": 0.08,
            "Ip": 0.06, "g": 9.81 }

        self.parameters = {}
        self.equations = "None"
        self.ct = 0
        self.parameters = default_bot
        self.p = None
        self.q = None
        self.force = None


    def set_parameter(self, name, value):
        self.parameters[name] = value

    def set_parameters(self, params):
        for key, value in params.items():
            try:
                self.parameters[key] = value
            except KeyError:
                pass

    def get_parameter(self, name):
        if name in self.parameters:
            return self.parameters[name]
        else:
            raise SysNoParameterError

    def set_IC(self, coord):
        self.q = coord 

    def get_IC(self):
        return self.q

    def set_force(self, F):
        self.force = F

    def solveDE(self, dt, x, f):
        tspan = (self.ct, self.ct + dt)
        sol =  odeint(f, x, tspan)
        return sol

    def update_current_state(self, dt, F = None):
        #print(self.force)
        if F is None:
            F = self.force
        
        # Update dynamics
        #self.dq = rk4(lambda t, q: self.vdq(t, q, F))
        #self.ct, self.q = self.ct + dt,  self.q + self.dq( self.ct, self.q, dt )

        #self.convert_sys()

        # Update kinematics
        #self.dp = rk4(lambda t, q: self.vdp(t, q, self.q))
        #self.p = self.p + self.dp( self.ct, self.p, dt )
        
        sol = self.solveDE(dt, self.q, lambda t, q: self.vdq(q, t, F))
        self.q = sol[-1, :]

        self.convert_sys()

        sol = self.solveDE(dt, self.p, lambda t, q: self.vdp(q, t, self.q))
        self.p = sol[-1, :]

        self.ct += dt

    def get_position_coordinates(self):
        return self.p
    
    def vdp(self, t, q, ic):
        raise NotImplementedError

    def vdq(self, t, q, F):
        raise NotImplementedError

    def convert_sys(self):
        pass

    def reset_time(self):
        self.ct = 0

    def __str__(self):
        s = 'System Model '
        s += type(self).__name__
        s += '\n'
        s += 'Equations: %s\n' % self.equations
        s += 'Inputs: %d\n' % len(self.force)
        s += 'Dynamics Space: %d\n' % len(list(self.q))
        s += 'Kinematics Space: %d\n' % len(list(self.p))
        return s