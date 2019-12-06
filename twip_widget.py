''' TWIP Viewer Widget

Ethan Lew
1/30/2019
elew@pdx.edu


This file contains a pyqtgraph implementation viewer of the TWIPZi object in the twip.py script. Also, in the __main__
only section, it also contains an example of how QT's timer system can create a time step controller to view the TWIP
in realtime. This effort is NOT complete by any means, needing:

    1.  Better coordinate manipulations than the one implemented in assemble_robot and draw_robot
    2.  Better parametric mesh editing, as the robot generated with extreme parameters look excessively 
        goofy and non-sensical. Better care when applying scale factor would help greatly.
    3. More in viewport visualization. Live traces of the robots path would be useful. Also, displaying
        quantity vectors like torque applied and other parameters on the TWIP itself would better commuinicate 
        the system state. 

Naming Conventions:
    <p/c>state - <previous/current> model state
    item_<name> - meshItem object
    mesh_<name> - gl.MeshData object
'''

from PyQt5 import QtGui, QtWidgets, QtOpenGL, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pywavefront as wave
from twip import TWIPZi, wraptopi
import numpy as np
from numpy import sin, cos, floor
import time
import OpenGL

def obj_to_mesh(obj_context, object_name):
    ''' Take geometry contents from a pywavefront scene and put it in a gl.MeshData structure
    '''
    faces = np.array(obj_context.meshes[object_name].faces)
    verts =  np.array(obj_context.vertices)
    md = gl.MeshData(vertexes=verts, faces=faces)
    return md

class TWIPWidget(gl.GLViewWidget):
    ''' TWIPWidget draws a TWIP robot in a 3D OpenGL PyQtGraph Widget Object

    TWIPWidget serves a view of a TWIP model, with no responsibility to update the system
    over time. As such, it has no ability to understand velocity or timing.
    Every coordinate given to it is positional, either as an angular or translational
    parameter. However, it is responsible for generating the correct shape,
    as specified by the dimension parameters:

    l - distance from the track to the COG
    d - length of the track
    r - radius of the wheel

    It also supports the model of LOCAL transforms -- the robot must be assembled in space
    before its planar motions can be described as a single mass. 
    '''
    def __init__(self, parent, twip, hud = True):
        gl.GLViewWidget.__init__(self, parent)

        # Import TWIP geometry from twip.obj
        twip_scene = wave.Wavefront('assets/twip.obj',  encoding="iso-8859-1", parse=True, collect_faces=True)
        self.mesh_wheel_r = obj_to_mesh(twip_scene, 'wheel_r')
        self.mesh_wheel_l = obj_to_mesh(twip_scene, 'wheel_l')
        self.mesh_track = obj_to_mesh(twip_scene, 'track')
        self.mesh_pendulum = obj_to_mesh(twip_scene, 'pendulum')
        self.mesh_load = obj_to_mesh(twip_scene, 'load')

        # get frequent parameters
        self.l = twip.get_parameter('l')
        self.d = twip.get_parameter('d')
        self.r = twip.get_parameter('r')

        self.twip = twip

        # setup grid
        g = gl.GLGridItem()
        g.scale(1,1,1)
        self.addItem(g)

        self.assemble_robot()

        self.pstate = self.twip.get_position_coordinates()

        
        self.painter = QtGui.QPainter()

        hud_font =  QtGui.QFont()
        hud_font.setPointSize(8)
        hud_font.setBold(True)
        hud_font.setWeight(75)

        self.hud_font = hud_font
        self.setAutoFillBackground(False)
        self.painter.setFont(self.hud_font)

        self.ptime = time.time()
        self.frames = 8
        self.cframe = 0
        self.fps = 0

        self.do_hud = hud

        
        

    def assemble_robot(self):
        ''' Create meshItems containing the robot's geometry and set mesh attributes, namely shading and parents
        '''
        gl_options = "opaque"
        shader = "edgeHilight"
        sf = 3
        wheel_color = (.8/sf, .8/sf, .8/sf, 0.2)
        load_color = wheel_color
        base_color = wheel_color

        # Load meshes as GLMeshItems
        self.item_wheel_l = gl.GLMeshItem(meshdata=self.mesh_wheel_l, smooth=True, color=wheel_color, shader=shader, glOptions=gl_options)
        self.item_wheel_r = gl.GLMeshItem(meshdata=self.mesh_wheel_r, smooth=True, color=wheel_color, shader=shader, glOptions=gl_options)
        self.item_load = gl.GLMeshItem(meshdata=self.mesh_load, smooth=False, color=load_color, shader=shader, glOptions=gl_options)
        self.item_pendulum = gl.GLMeshItem(meshdata=self.mesh_pendulum, smooth=False, color=base_color, shader=shader, glOptions=gl_options)
        self.item_track = gl.GLMeshItem(meshdata=self.mesh_track, smooth=True, color=base_color, shader=shader, glOptions=gl_options)
        
        # Establish Parent Child Relationships
        self.item_wheel_l.setParent(self.item_track)
        self.item_wheel_r.setParent(self.item_track)
        self.item_pendulum.setParent(self.item_track)
        self.item_load.setParent(self.item_pendulum)


        self.draw_origin()
        
        # Add to scene
        self.addItem(self.item_wheel_l)
        self.addItem(self.item_wheel_r)
        self.addItem(self.item_load)
        self.addItem(self.item_pendulum)
        self.addItem(self.item_track)

    def draw_origin(self):
        ''' Set local transformations to draw an assembled TWIP at the origin
        '''
        # Scale Right Wheel
        self.item_wheel_r.scale(self.r, self.r, self.r, local=True)
        self.item_wheel_r.translate((self.d/2 - 1.5*self.r), 0, 0)


        # Scale Left Wheel
        self.item_wheel_l.scale(self.r, self.r, self.r, local=True)
        self.item_wheel_l.translate((-self.d/2 + 1.5*self.r), 0, 0)

        # Scale Track
        self.item_track.scale(self.d / 3, self.r, self.r, local=True)

        # Scale Pendulum
        self.item_pendulum.scale(self.r, self.r, self.l, local=True)

        # Move Load
        self.item_load.scale(self.r, self.r, 1, local=True)
        self.item_load.translate(0, 0, 2*(self.l - 1), local=True)

        self.translate_all(0, 0, self.r)

    def draw_twip(self, D=None):
        ''' Given a TWIP's positional coordinates, draw a TWIP on canvas with that orientation.
        '''

        # Get moving average FPS
        if self.fps == 0:
            ctime = time.time()
            self.fps = (1/(ctime - self.ptime))
            self.ptime = ctime
        if self.cframe == self.frames:
            ctime = time.time()
            self.fps = (self.frames/(ctime - self.ptime))
            self.ptime = ctime
            self.cframe = 0

        self.cframe = self.cframe + 1

        rad_to_deg = 180.0/3.141592654 

        cstate = self.twip.get_position_coordinates() 

        self.rotate_all(-90, 0, 0, 1)
        self.translate_all(-self.pstate[0], -self.pstate[1], 0)
        self.translate_all(0, 0, -self.r)

        self.rotate_all(90, 0, 0, 1)
        self.rotate_payload(-self.pstate[5]*rad_to_deg, cos(self.pstate[2]), sin(self.pstate[2]), 0)
        self.rotate_all(-90, 0, 0, 1)

        self.rotate_all(-self.pstate[2]*rad_to_deg, 0, 0, 1)

        self.rotate_all(90, 0, 0, 1)
        self.item_wheel_r.rotate((self.pstate[3]*rad_to_deg), 1, 0, 0)
        self.item_wheel_l.rotate((self.pstate[4]*rad_to_deg), 1, 0, 0)

        self.item_wheel_r.rotate(-(cstate[3]*rad_to_deg), 1, 0, 0)
        self.item_wheel_l.rotate(-(cstate[4]*rad_to_deg), 1, 0, 0)
        self.rotate_all(-90, 0, 0, 1)

        self.rotate_all(cstate[2]*rad_to_deg, 0, 0, 1)

        self.rotate_all(90, 0, 0, 1)
        self.rotate_payload(cstate[5]*rad_to_deg, cos(cstate[2]), sin(cstate[2]), 0)
        self.rotate_all(-90, 0, 0, 1)

        self.translate_all(0, 0, self.r)
        self.translate_all(cstate[0], cstate[1], 0)

        self.rotate_all(90, 0, 0, 1)

        self.pstate = cstate
        try:
            self.paintGL()
        except ZeroDivisionError:
            pass
        except OpenGL.error.Error:
            pass

    def set_c(self, c):
        self.c = c
        
    def paintGL(self, *args, **kwargs):
        ''' Redefine the paintGL method to include painter
        '''
        gl.GLViewWidget.paintGL(self, *args, **kwargs)

        if self.do_hud:
            self.paint_hud()


    def paint_hud(self, D = None):
        ''' Paint the HUD information
        '''
        w, h = self.width(), self.height()
        self.painter.begin(self)
        self.painter.setPen(QtCore.Qt.white)
        self.painter.setFont(self.hud_font)
        self.painter.drawText(QtCore.QRectF(3, 3,w,h), QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop,  "%s\nα: %3.3f\nθ: %3.3f \n(x, y): %1.3f, %1.3f" % (self.twip.equations, wraptopi(self.pstate[5])*180/3.141592, wraptopi(self.pstate[2])*180/3.141592, self.pstate[0], self.pstate[1]))
        self.painter.drawText(QtCore.QRectF(0,0,w - 3,h - 3), QtCore.Qt.AlignRight|QtCore.Qt.AlignBottom, "%d FPS" % round(self.fps)  )
        #self.painter.drawText(QtCore.QRectF(0,0,w - 3,h - 3), QtCore.Qt.AlignRight|QtCore.Qt.AlignTop, "Controller: PID\nP: %1.3f\nD: %1.3f\nI: %1.3f" % (self.c.kp, self.c.kd, self.c.ki)  )
        self.painter.end()

    def translate_all(self, x, y, z, local=False):
        ''' Helper function to move all meshItems
        '''
        self.item_wheel_l.translate(x, y, z, local=local)
        self.item_wheel_r.translate(x, y, z, local=local)
        self.item_load.translate(x, y, z, local=local)
        self.item_pendulum.translate(x, y, z, local=local)
        self.item_track.translate(x, y, z, local=local)

    def rotate_all(self, angle, x, y, z, local=False):
        ''' Helper function to rotate all meshItems
        '''
        self.item_wheel_l.rotate(angle, x, y, z, local=local)
        self.item_wheel_r.rotate(angle, x, y, z, local=local)
        self.item_load.rotate(angle, x, y, z, local=local)
        self.item_pendulum.rotate(angle, x, y, z, local=local)
        self.item_track.rotate(angle, x, y, z, local=local)

    def rotate_payload(self, angle, x, y, z, local=False):
        ''' Helper function to rotate everything except the wheels
        '''
        self.item_load.rotate(angle, x, y, z, local=local)
        self.item_pendulum.rotate(angle, x, y, z, local=local)
        self.item_track.rotate(angle, x, y, z, local=local)

if __name__ == "__main__":
    class MainWindow(QtWidgets.QMainWindow):
        ''' Realtime TWIP viewer program
        '''
        def __init__(self):
            super(MainWindow, self).__init__()

            # Create TWIP model
            self.twip = TWIPZi()
            self.twip_widget = TWIPWidget(self, self.twip)

            # Add layout to put twip_widget in
            wid = QtWidgets.QWidget(self)
            self.setCentralWidget(wid)
            mainLayout = QtWidgets.QHBoxLayout()
            mainLayout.addWidget(self.twip_widget)
            wid.setLayout(mainLayout)

            # Setup twip initial state
            dt = 1/30
            self.twip.set_IC([0, 0, 0, 0, 0, 0])
            self.twip.update_current_state(dt, [1/dt*0.1, -1/dt*0.1,  0, 0]) 
            self.dt = dt
            
        def update_twip(self):
            ''' program mainloop method
            '''
            self.twip.update_current_state(self.dt, [0, 0,  0, 0])
            self.twip_widget.draw_twip()

    app = QtWidgets.QApplication(['TWIP Viewer'])
    window = MainWindow()
    window.resize(200, 200)
    
    sim_timer = QtCore.QTimer()
    sim_timer.timeout.connect(window.update_twip)
    sim_timer.start(1/90*1000)

    window.show()

    app.exec_()

