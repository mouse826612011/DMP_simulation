# -*- coding: utf-8 -*-
"""
This code is a demonstration-getting program of Emika-Panda robotic arm based on Pybullet.

Author: Cheems_JH
Date: 2023.18
"""

import pybullet as p
import pybullet_data as pd
import math
import time
import openpyxl
import datetime


# ------ Function: draw target circle
def draw_area(surface, c_x, c_y, w, color):
    from_s = [[c_x+w/2, c_y+w/2, 0], [c_x+w/2, c_y+w/2, 0], [c_x-w/2, c_y+w/2, 0], [c_x+w/2, c_y-w/2, 0]]
    to_s = [[c_x+w/2, c_y-w/2, 0], [c_x-w/2, c_y+w/2, 0], [c_x-w/2, c_y-w/2, 0], [c_x-w/2, c_y-w/2, 0]]
    for f, t in zip(from_s, to_s):
        surface.addUserDebugLine(lineFromXYZ=f, lineToXYZ=t, lineColorRGB=color, lineWidth=2)


# ------ Connect to the pybullet server and load the environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
# load models
pandaUid = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
tableUid = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.65])
table2Uid = p.loadURDF("table/table.urdf", basePosition=[0.5, -1, -0.65])
table3Uid = p.loadURDF("table/table.urdf", basePosition=[0.5, 1, -0.65])
trayUid = p.loadURDF("tray/traybox.urdf", basePosition=[1, 0, 0],
                     baseOrientation=[0, 1, 0, 0], globalScaling=0.5)
objectUid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.7, 0, 0.2])
object2Uid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.4, 0, 0.2])
object3Uid = p.loadURDF("random_urdfs/000/000.urdf", basePosition=[0.95, 0, 0.2])
# set gravity and camera
p.setGravity(0, 0, -9.81)
p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=0,
                             cameraPitch=-45, cameraTargetPosition=[0.55, -0.35, 0.2])
# draw the target area
draw_area(p, 0.5, 0.5, 0.2,[255, 255, 255])
draw_area(p, 0.5, -0.5, 0.2,[255, 0, 0])
# create sliders to control the robotic arm's joints and finger
joint_sliders = []
for i in range(7):
    slider = p.addUserDebugParameter(f"Joint {i+1}", -math.pi, math.pi, 0)
    joint_sliders.append(slider)
finger_slider = p.addUserDebugParameter("finger", 0, 0.04, 0)

# ------ Initial parameters
finger = 0
joint_index = 7
T = 0.1  # sampling time interval, in seconds
last_time = time.time()
initial_angles = p.calculateInverseKinematics(pandaUid, joint_index, [0.5, 0.5, 0.5], [0, 0, 0, -1])
jointAngles = list(initial_angles)
data_list = []

# ------ Simulation start
while True:
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
    p.stepSimulation()
    # press q to break simulation
    keys = p.getKeyboardEvents()
    if ord("q") in keys:
        break

    # read data from the slider and control the robotic arm
    for i in range(7):
        jointAngles[i] = p.readUserDebugParameter(joint_sliders[i])  # 从滑块读取关节角度
        p.setJointMotorControl2(pandaUid, i, p.POSITION_CONTROL, jointAngles[i])
    finger = p.readUserDebugParameter(finger_slider)
    p.setJointMotorControl2(pandaUid, 9, p.POSITION_CONTROL, finger)
    p.setJointMotorControl2(pandaUid, 10, p.POSITION_CONTROL, finger)

    # get pos of the target joint in T
    current_time = time.time()
    if current_time - last_time >= T:
        # print("Time时间戳:", current_time - last_time)
        last_time = current_time
        # get joint state
        link_state = p.getLinkState(pandaUid, joint_index)
        link_position = link_state[0]
        link_orientation = link_state[1]
        # print("Joint索引:", joint_index)
        # print("Link位置:", link_position)
        # print("Link方向:", link_orientation)
        # append data in data list
        data_list.append([current_time, *link_position, *link_orientation, finger])

# ------ Processing after Quit the simulation
p.disconnect()
wb = openpyxl.Workbook()  # initial excel
ws = wb.active
# writing headers
ws.append(["Time", "Pos_x", "Pos_y", "Pos_z",
           "Pos_q1", "Pos_q2", "Pos_q3", "Pos_q4", "finger"])
# input excel
for data in data_list:
    ws.append(data)
# save .Excel
dt = datetime.datetime.now()
dt_str = dt.strftime("%Y-%m-%d_%H-%M-%S")
excel_filename = "demo" + dt_str + ".xlsx"
wb.save(excel_filename)
print("Demonstration Over")
