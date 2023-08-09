# -*- coding: utf-8 -*-
"""
This code is a 2D simulation of imitation learning for a two-link robotic arm based on DMP.
In this program, the user will draw a demonstration trajectory using the mouse,
and the trained DMP will drive the robotic arm to reproduce or generalize the trajectory.

Author: Cheems_JH
Date: 2023.8.9
"""
import numpy as np
import pygame
import matplotlib.pyplot as plt
import sys
import discrete_dmp_1D
from scipy.interpolate import interp1d
import math

# ------ User-adjustable parameters (suggested)
k_interpolation = 4  # multiple of interpolation, integer greater than or equal to 1
n_rbf = 500  # numbers of DMP basis functions, integer
start_bias_x = -40  # generalization in space and scale can be achieved by adjusting the start and end points
start_bias_y = 40
target_bias_x = 40
target_bias_y = -40
tau = 1.0  # scaling in time


# ------ Class: update and plot the state and auxiliary information of the manipulator
class RoboticArm:
    def __init__(self, o_x=200, o_y=450, d1=300, d2=200, theta1=np.pi/4, theta2=0):
        self.o_x = o_x
        self.o_y = o_y
        self.d1 = d1
        self.d2 = d2
        self.theta1 = theta1
        self.theta2 = theta2
        self.l_axis = 400
        self.actuator_history = []
        self.black = (0, 0, 0)
        self.w_arm = 2
        self.r_joint = 5
        self.arm1_color = (0, 0, 255)
        self.arm2_color = (255, 0, 0)
        self.history_color = (255, 127, 80)
        self.history_w = 3

    # ------ Function: transform the coordinate system
    def to_cartesian(self, x, y):
        to_x = self.o_x + x
        to_y = self.o_y - y
        return to_x, to_y

    # ------ Function: update system state
    def update_state(self, theta1=None, theta2=None):
        if theta1 is not None:
            self.theta1 = theta1
        if theta2 is not None:
            self.theta2 = theta2

    # ------ Function: draws the current frame of the simulation
    def draw_frame(self, surface):
        # draw X-axis
        arrow_size = 10
        start_arrow = (0, 0)
        end_arrow = (0, self.l_axis)
        x_arrow_left = (end_arrow[0]-arrow_size*np.sin(np.pi/4), end_arrow[1]-arrow_size*np.sin(np.pi/4))
        x_arrow_right = (end_arrow[0]+arrow_size*np.sin(np.pi/4), end_arrow[1]-arrow_size*np.sin(np.pi/4))
        pygame.draw.line(surface, self.black, self.to_cartesian(start_arrow[0], start_arrow[1]),
                         self.to_cartesian(end_arrow[0], end_arrow[1]), 2)
        pygame.draw.line(surface, self.black, self.to_cartesian(end_arrow[0], end_arrow[1]),
                         self.to_cartesian(x_arrow_left[0], x_arrow_left[1]), 2)
        pygame.draw.line(surface, self.black, self.to_cartesian(end_arrow[0], end_arrow[1]),
                         self.to_cartesian(x_arrow_right[0], x_arrow_right[1]), 2)
        # draw Y-axis
        end_arrow = (self.l_axis, 0)
        y_arrow_left = (end_arrow[0] - arrow_size * np.sin(np.pi / 4), end_arrow[1] + arrow_size * np.sin(np.pi / 4))
        y_arrow_right = (end_arrow[0] - arrow_size * np.sin(np.pi / 4), end_arrow[1] - arrow_size * np.sin(np.pi / 4))
        pygame.draw.line(surface, self.black, self.to_cartesian(start_arrow[0], start_arrow[1]),
                         self.to_cartesian(end_arrow[0], end_arrow[1]), 2)
        pygame.draw.line(surface, self.black, self.to_cartesian(end_arrow[0], end_arrow[1]),
                         self.to_cartesian(y_arrow_left[0], y_arrow_left[1]), 2)
        pygame.draw.line(surface, self.black, self.to_cartesian(end_arrow[0], end_arrow[1]),
                         self.to_cartesian(y_arrow_right[0], y_arrow_right[1]), 2)

        # draw actuator history
        if len(self.actuator_history) > 0:
            for pos in self.actuator_history:
                pygame.draw.circle(surface, self.history_color, self.to_cartesian(pos[0], pos[1]), self.history_w)

        # draw arm1
        p1 = (self.d1*np.cos(self.theta1), self.d1*np.sin(self.theta1))
        pygame.draw.line(surface, self.arm1_color, self.to_cartesian(start_arrow[0], start_arrow[1]),
                         self.to_cartesian(p1[0], p1[1]), self.w_arm)
        pygame.draw.circle(surface, self.arm1_color, self.to_cartesian(start_arrow[0], start_arrow[1]), self.r_joint)
        # draw arm2
        pygame.draw.circle(surface, self.arm2_color, self.to_cartesian(p1[0], p1[1]), self.r_joint)
        p2 = (p1[0]+self.d2*np.cos(self.theta1+self.theta2), p1[1]+self.d2*np.sin(self.theta1+self.theta2))
        pygame.draw.line(surface, self.arm2_color, self.to_cartesian(p1[0], p1[1]),
                         self.to_cartesian(p2[0], p2[1]), self.w_arm)
        pygame.draw.circle(surface, self.arm2_color, self.to_cartesian(p2[0], p2[1]), self.r_joint)

        # store executor data
        self.actuator_history.append(p2)


# ------ Initial simulation
pygame.init()
width, height = 800, 600  # set window's size
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Simulation: DMP-based LfD for two-link robotic arm")  # set title
# parameters about the demonstration
mouse_track = []
mouse_color = (153, 204, 255)
mouse_left_down = False
# parameters about the robotic arm
angle1 = 0
angle2 = 0
robot = RoboticArm(theta1=angle1, theta2=angle2)
# parameters about the simulation status
font = pygame.font.Font(None, 20)
current_text = "STATUS: Press 'Enter' to start the demonstration..."
text_rect = None  # position of text
simulation_state = 'START'
# parameters about the DMP training
x_track = []
y_track = []
dmp_x = []
dmp_y = []
x_re = []
y_re = []
x_re_robot = []
y_re_robot = []
interpolate_x = []
interpolate_y = []
# parameters about the robot reproduction
cnt = 0

running = True
# ------ Simulation start
while running:
    # ------ Event trigger
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif simulation_state == 'START' and event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:  # 回车按下
            current_text = "STATUS: Keep pressing the left mouse button to demonstrate..."
            simulation_state = 'GET-DEMO'
        elif simulation_state == 'GET-DEMO' and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # 左键按下
            mouse_left_down = True
            demo_track = []
        elif simulation_state == 'GET-DEMO' and event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # 左键松开
            mouse_left_down = False
            simulation_state = 'TRAINING'
            current_text = "STATUS: Demo has been obtained, in DMP training..."
            continue
    screen.fill((255, 255, 255))

    # ------ Get demonstration when in 'GET-DEMO' state
    if simulation_state == 'GET-DEMO' and mouse_left_down:
        # get mouse position
        mouse_pos = pygame.mouse.get_pos()
        mouse_track.append(mouse_pos)
        # draw them
        if len(mouse_track) >= 2:
            pygame.draw.lines(screen, mouse_color, False, mouse_track)

    # ------ Training DMP in 'TRAINING' state
    if simulation_state == 'TRAINING':
        x_track = np.zeros(len(mouse_track))
        y_track = np.zeros(len(mouse_track))
        k = 0
        for mouse in mouse_track:
            demo_x = mouse[0] - 200
            demo_y = 450 - mouse[1]
            x_track[k] = demo_x
            y_track[k] = demo_y
            k = k + 1
        # training and reproduction
        dmp_x = discrete_dmp_1D.DiscreteDMP(data_set=x_track, n_rbf=n_rbf)
        dmp_y = discrete_dmp_1D.DiscreteDMP(data_set=y_track, n_rbf=n_rbf)
        x_re, dx_re, ddx_re = dmp_x.reproduction(start=(x_track[0]+start_bias_x), target=(x_track[-1]+target_bias_x),
                                                 tau=tau)
        y_re, dy_re, ddy_re = dmp_y.reproduction(start=(y_track[0]+start_bias_y), target=(y_track[-1]+target_bias_y),
                                                 tau=tau)
        # mapping the DMP trajectory is onto the robot coordinate system
        x_re_robot = np.zeros(len(x_re))
        y_re_robot = np.zeros(len(y_re))
        for i in range(len(x_re)):
            x_re_robot[i] = x_re[i]
        for j in range(len(y_re)):
            y_re_robot[j] = y_re[j]
        # interpolate reproduction trajectory as need
        n_data_o = np.linspace(0, len(x_re_robot), len(x_re_robot))
        target_n = len(x_re_robot) * k_interpolation
        n_new = np.linspace(0, len(x_re_robot), target_n)
        # interpolate
        interp_func_x = interp1d(n_data_o, x_re_robot, kind='cubic')
        interpolate_x = interp_func_x(n_new)
        interp_func_y = interp1d(n_data_o, y_re_robot, kind='cubic')
        interpolate_y = interp_func_y(n_new)
        # updating the simulation state
        simulation_state = 'REPRODUCTION'
        current_text = "STATUS: the DMP training is over. Start to reproduce..."

    # Calculate the inverse kinematics in 'REPRODUCTION' state
    if simulation_state == 'REPRODUCTION':
        cos_angle2 = ((pow(interpolate_x[cnt], 2) + pow(interpolate_y[cnt], 2) - pow(robot.d1, 2) - pow(robot.d2, 2))
                      / (2 * robot.d1 * robot.d2))
        if cos_angle2 > 1 or cos_angle2 < -1:
            print('Warning: unable to reach point.')
        angle2 = math.acos(cos_angle2)
        angle1 = (math.atan2(interpolate_y[cnt], interpolate_x[cnt]) -
                  math.atan2(robot.d2*math.sin(angle2), robot.d1+robot.d2*math.cos(angle2)))
        robot.update_state(theta1=angle1, theta2=angle2)
        # terminal condition
        if cnt == (len(interpolate_x)-1):
            simulation_state = 'OVER'
            current_text = "STATUS: reproduction is over. Thanks."
            continue
        cnt = cnt + 1

    # ------ Draw frames
    # 1. draw demonstration
    if simulation_state != 'GET-DEMO' and len(mouse_track) >= 2:
        pygame.draw.lines(screen, mouse_color, False, mouse_track, width=5)
    # 2. draw robotic arm
    robot.draw_frame(surface=screen)
    # 3. draw status text
    text_surface = font.render(current_text, True, (0, 0, 0))
    if text_rect is None:
        text_rect = text_surface.get_rect(left=width/3, centery=height // 15)
    screen.blit(text_surface, text_rect)
    # render current frame
    pygame.display.flip()
    # set FPS
    pygame.time.Clock().tick(100)

# ------ Plot DMP training results
plt.figure(figsize=(10, 5))
plt.plot(x_track, 'g', label='demo_x')
plt.plot(x_re, 'b', label='repr_x')
plt.plot(y_track, 'r', label='demo_y')
plt.plot(y_re, 'k', label='repr_y')
plt.legend()
plt.grid()
plt.xlabel('time')
plt.ylabel('value')
plt.title('The DMP training results in the world coordinate system')
plt.show()

# ------ Quit simulation
pygame.quit()
sys.exit()
