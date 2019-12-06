''' TWIP Robot

Ethan Lew
4/18/19
elew@pdx.edu

Model complete robot dynamics, including controllers
'''

from pid import IterPID
from motor import PWMDCMotor
from twip import TWIPZi
from system import SysBase

from PyQt5 import QtGui, QtWidgets, QtOpenGL, QtCore
from twip_widget import TWIPWidget

import numpy as np

class PIDRobot(SysBase):
    def __init__(self, Ts):
        self.sp_tilt = 0
        self.sp_yaw = 0

        self.twip = TWIPZi()
        self.equations = self.twip.equations

        self.motor_l = PWMDCMotor()
        self.motor_r = PWMDCMotor()
        self.motor_l.set_IC([0, 0])
        self.motor_r.set_IC([0, 0])

        self.pid_tilt = IterPID(Ts)
        self.pid_yaw = IterPID(Ts)

        self.pid_tilt.tune(70, 1, 300)
        self.pid_yaw.tune(25, 2, 0)
        
        self.pid_tilt.set_IC([0, 0, 0, 0])
        self.pid_yaw.set_IC([0, 0, 0, 0])

        self.parameters = self.twip.parameters

    def update_current_state(self, dt, F =None):
        # get twip state
        coords = self.twip.get_position_coordinates()
        curr_yaw = coords[2]
        curr_tilt = coords[5]
        
        #print(coords)
        # Get error signals
        err_y = self.sp_yaw - curr_yaw
        err_t = self.sp_tilt - curr_tilt
        #print(err_t)

        # Update PID
        self.pid_yaw.update_current_state(dt, [err_y])
        self.pid_tilt.update_current_state(dt, [err_t])

        # Get PID values
        ctrl_t = self.pid_tilt.get_position_coordinates()
        ctrl_y = self.pid_yaw.get_position_coordinates()

        # Convert to PWM values
        pwm_t = int(ctrl_t)
        pwm_y = int(ctrl_y)

        #print(pwm_t, pwm_y)

        # Update Motors
        self.motor_l.update_current_state(dt, [-pwm_t + pwm_y])
        self.motor_r.update_current_state(dt, [-pwm_t - pwm_y])

        

        # Get motor torques
        t_l = self.motor_l.get_position_coordinates()[0]
        t_r = self.motor_r.get_position_coordinates()[0]

        tF = [t_l*10, t_r*10,  0, 0]


        if F is not None:
            tF = [(tF[i] + F[i]) for i in range(0, len(F))] 

        # Update TWIP
        self.twip.update_current_state(dt, tF)

    def set_tilt(self, tilt):
        self.sp_tilt = tilt

    def set_yaw(self, yaw):
        self.sp_yaw = yaw

    def set_IC(self, coord):
        self.twip.set_IC(coord)

    def get_position_coordinates(self):
        return self.twip.get_position_coordinates()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from plot_widget import RollingPlotWidget
    class MainWindow(QtWidgets.QMainWindow):
        ''' Realtime TWIP viewer program
        '''
        def __init__(self):
            super(MainWindow, self).__init__()

            # Create TWIP model
            self.twip = PIDRobot(0.03)
            self.twip_widget = TWIPWidget(self, self.twip)

            self.resize(1900,1000)
            # Add layout to put twip_widget in
            wid = QtWidgets.QWidget(self)
            self.setCentralWidget(wid)

            # create layouts
            self.view_layout =  QtGui.QHBoxLayout()
            self.twip_layout = QtGui.QHBoxLayout()
            self.plot_layout = QtGui.QVBoxLayout()

            wid.setLayout(self.view_layout)
            self.view_layout.addLayout(self.twip_layout, 2)
            self.view_layout.addLayout(self.plot_layout, 1)

            self.twip_layout.addWidget(self.twip_widget)

            self.tilt_widget = RollingPlotWidget(1, 300)
            self.yaw_widget = RollingPlotWidget(1, 300)
            self.motor_plot_widget = RollingPlotWidget(2, 300)

            self.tilt_widget.set_pen(0, 'r')
            self.tilt_widget.setLabel( 'left', 'Tilt', units='degrees')
            self.tilt_widget.setLabel('bottom', 'Sample Number')
            self.tilt_widget.showGrid(True, True, 0.5)

            self.yaw_widget.set_pen(0, 'c')
            self.yaw_widget.setLabel( 'left', 'Yaw', units='degrees')
            self.yaw_widget.setLabel( 'bottom', 'Sample Number')
            self.yaw_widget.showGrid(True, True, 0.5)

            self.motor_plot_widget.set_pen(0, 'g')
            self.motor_plot_widget.set_pen(1, 'w')
            self.motor_plot_widget.setLabel('left', 'Motor Torque', units='N m')
            self.motor_plot_widget.setLabel( 'bottom', 'Sample Number')
            self.motor_plot_widget.showGrid(True, True, 0.5)

            self.plot_layout.addWidget(self.tilt_widget)
            self.plot_layout.addWidget(self.yaw_widget)
            self.plot_layout.addWidget(self.motor_plot_widget)

            #wid.setLayout(mainLayout)

            # Setup twip initial state
            dt = 1/30
            self.twip.set_IC([0, 0, 0.01, 0.0, 0, 0])
            #self.twip.update_current_state(dt, [1/dt*0.5, 1/dt*0.4,  0, 0]) 
            self.dt = dt
            
        def update_twip(self):
            ''' program mainloop method
            '''
            self.twip.update_current_state(self.dt, [0, 0,  0, 0])

            m_l = self.twip.motor_l.get_position_coordinates()[0]
            m_r = self.twip.motor_r.get_position_coordinates()[0]
            self.motor_plot_widget.push_data([m_l, m_r])

            coords = self.twip.get_position_coordinates()
            y = coords[2]*180/3.1415
            t = coords[5]*180/3.1415
            self.tilt_widget.push_data([t])
            self.yaw_widget.push_data([y])

        def update_plot(self):
            #pass
            self.motor_plot_widget.update_plot()
            self.tilt_widget.update_plot()
            self.yaw_widget.update_plot()
            self.twip_widget.draw_twip()

    app = QtWidgets.QApplication(['TWIP Viewer'])
    window = MainWindow()
    
    sim_timer = QtCore.QTimer()
    sim_timer.timeout.connect(window.update_twip)
    sim_timer.start(1/90*1000)

    plot_timer = QtCore.QTimer()
    plot_timer.timeout.connect(window.update_plot)
    plot_timer.start(1/40*1000)

    window.show()

    app.exec_()


    
