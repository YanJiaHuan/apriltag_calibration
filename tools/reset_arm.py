import numpy as np
import time
import json
from pathlib import Path
import sys
from third_party.RM_API2.Python.Robotic_Arm.rm_robot_interface import *

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("10.168.1.18", 8080)

reset_joints_deg = [35.635, -41.329, -99.03, -62.103, 71.107, 154.35]
ret = arm.rm_movej(reset_joints_deg, 5, 0, 0, 1) #速度比例，交融半径百分比系数，轨迹连接标志，阻塞设置（1为阻塞）
time.sleep(2.0)

arm.rm_delete_robot_arm()



