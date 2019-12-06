from robot import PIDRobot

from PyQt5 import QtGui, QtWidgets, QtOpenGL, QtCore
from twip_widget import TWIPWidget

import numpy as np
import random

def is_stable(IC, t_max = 5, a_max = 3.1415/180*80, stable_box = [0, 0, 0]):
    # Setup twip initial state
    twip = PIDRobot(0.03)
    dt = 1/30
    twip.set_IC(IC)
    twip.update_current_state(dt, [0, 0,  0, 0]) 

    t = 0
    while(t < t_max):    
        twip.update_current_state(dt, [0, 0,  0, 0])
        t += dt
        cpos = twip.twip.q
        if((abs(cpos[2]) < stable_box[0]) and (abs(cpos[5]) < stable_box[1]) and (abs(cpos[3]) < stable_box[2])):
            return True
        if(abs(cpos[5]) > 2):
            return False
    return True

def find_stable_box(err=0.05):
    # Find a stable box
    norm_err = 1000
    err_tol = err
    box = np.array([10, 10, 10])
    last_box = box*0
    stab = False
    while((norm_err > err_tol) or not stab):
        stab = is_stable([0, 0, box[0], box[2], 0, box[1]])      
        if(stab):
            box = box + abs(last_box - box)/2
            norm_err = np.linalg.norm(abs(last_box - box))
            last_box = box
        else:
            box = box - abs(last_box - box)/2
    return box

stable_box = find_stable_box()
n_points = 1000
n_found = 0
a_range = [-np.pi/2, np.pi/2]
ad_range = [-3/2, 3/2]
v_range = [-10, 10]
stab_pts = np.zeros((1, 3))
while(n_found < n_points):
    pt = np.array([0, 0, random.uniform(a_range[0], a_range[1]),
                     random.uniform(v_range[0], v_range[1]), 0,
                        random.uniform(ad_range[0], ad_range[1])])

    if is_stable(pt, stable_box=stable_box):
        print(pt, ' is stable (%d/%d)' % (n_found, n_points))
        spt = np.array([[pt[2]], [pt[5]], [pt[3]]])
        stab_pts = np.append(stab_pts, spt.transpose(), axis=0)
        n_found += 1

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np

app = QtGui.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 20
w.show()
w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')

g = gl.GLGridItem()
w.addItem(g)

##
##  Third example shows a grid of points with rapidly updating position
##  and pxMode = False
##

print(stab_pts)
pos3 = stab_pts
sp3 = gl.GLScatterPlotItem(pos=pos3, color=(1,1,1,.7), size=0.1, pxMode=False)

w.addItem(sp3)

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()