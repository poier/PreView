"""
Copyright 2018 ICG, Graz University of Technology

This file is part of PreView.

PreView is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PreView is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PreView.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np


#%% Intrinsic calibration parameters
# gener8 ID ..-1221
fxCam1221=212.608
fyCam1221=212.608
cxCam1221=109.897
cyCam1221=85.4924
iCalibCam1221 = [fxCam1221, fyCam1221, cxCam1221, cyCam1221, 0.0611, -2.4763, -0.0018, 0.0001, 4.5695] # [fx, fy, cx, cy, k1, k2, p1, p2, k3]

# gener8 ID ..-1819
fxCam1819=214.06
fyCam1819=214.06
cxCam1819=110.942
cyCam1819=83.7219
iCalibCam1819 = [fxCam1819, fyCam1819, cxCam1819, cyCam1819, 0.0762, -2.6375, -0.0025, -0.0012, 4.8022] # [fx, fy, cx, cy, k1, k2, p1, p2, k3]

# gener8 ID ..-0819
fxCam0819=212.153
fyCam0819=212.153
cxCam0819=112.334
cyCam0819=81.9783
k1Cam0819=0.0615624
k2Cam0819=-2.42574
k3Cam0819=4.42566
p1Cam0819=-0.00294761
p2Cam0819=0.00162537
iCalibCam0819 = [fxCam0819, fyCam0819, cxCam0819, cyCam0819, k1Cam0819, k2Cam0819, p1Cam0819, p2Cam0819, k3Cam0819]

# gener8 ID ..-1617
fxCam1617=212.759
fyCam1617=212.759
cxCam1617=103.019
cyCam1617=85.055
k1Cam1617=0.0883007
k2Cam1617=-2.63266
k3Cam1617=4.78867
p1Cam1617=-0.00239052
p2Cam1617=-0.00121125
iCalibCam1617 = [fxCam1617, fyCam1617, cxCam1617, cyCam1617, k1Cam1617, k2Cam1617, p1Cam1617, p2Cam1617, k3Cam1617]

# PicoFlexx ..-0709
fxCam0709=214.433 
fyCam0709=214.433
cxCam0709=114.034 
cyCam0709=86.5364
k1Cam0709=0.099552 
k2Cam0709=-2.81809 
k3Cam0709=5.24012
p1Cam0709=0.00178 
p2Cam0709=-0.001747
iCalibCam0709 = [fxCam0709, fyCam0709, cxCam0709, cyCam0709, k1Cam0709, k2Cam0709, p1Cam0709, p2Cam0709, k3Cam0709]

# PicoFlexx ..-0515
fxCam0515=214.827 
fyCam0515=214.827
cxCam0515=116.544 
cyCam0515=86.6469
k1Cam0515=0.005981 
k2Cam0515=-2.26856 
k3Cam0515=4.3531
p1Cam0515=-0.001456 
p2Cam0515=0.000451
iCalibCam0515 = [fxCam0515, fyCam0515, cxCam0515, cyCam0515, k1Cam0515, k2Cam0515, p1Cam0515, p2Cam0515, k3Cam0515]


#%% Stereo calib
# gener8 ID 0005-1207-0034-1221 - 3.10.
R1_0310 = np.array([[-0.4988,   -0.3732,   -0.7692],
                  [-0.8544,    0.1844,    0.4645],
                  [0.0318,   -0.8980,    0.4150]])
t1_0310 = np.array([362.0231, -196.0407, 21.0797])
# gener8 ID 0005-1207-0034-1819 - 3.10.
R2_0310 = np.array([[0.5271,   -0.4205,    0.7356],
                  [-0.8453,   -0.2006,    0.4910],
                  [0.0590,    0.8824,    0.4622]])
t2_0310 = np.array([-364.3528, -199.2263, 20.1097])

# gener8 ID 0005-1207-0034-1819 - 16.11.
R1_1611 = np.array([[-0.5020,   -0.3819,   -0.7698],
                  [-0.8582,    0.1765,    0.4720],
                  [0.0446,   -0.9019,    0.4184]])
t1_1611 = np.array([363.6122, -198.0264,   20.2497])
# gener8 ID 0005-1207-0034-1221 - 16.11.
R2_1611 = np.array([[0.5131,   -0.3871,    0.7586],
                  [-0.8502,   -0.1819,    0.4823],
                  [0.0490,    0.8976,    0.4249]])
t2_1611 = np.array([-363.3488, -197.8591,   20.6216])

# gener8 ID 0005-1207-0034-1819 - 12.01.2017
R1_1201 = np.array([[-0.5056,   -0.2714,   -0.7991],
                  [-0.8438,    0.1832,    0.4716],
                  [-0.0187,   -0.9277,    0.3269]])
t1_1201 = np.array([360.0161, -194.8982,   22.5124])
# gener8 ID 0005-1207-0034-1221 - 12.01.2017
R2_1201 = np.array([[0.5359,   -0.2883,    0.7789],
                  [-0.8305,   -0.1912,    0.5006],
                  [-0.0046,    0.9259,    0.3459]])
t2_1201 = np.array([-361.2894, -196.4836,   21.7624])

# gener8 ID 0005-1207-0034-0819 - 06.02.2017
R1_0602 = np.array([[-0.4919,   -0.2992,   -0.8008],
                  [-0.8548,    0.1687,   0.4621],
                  [0.0032,   -0.9245, 0.3435]])
t1_0602 = np.array([360.7592, -195.5137,   21.9947])
# gener8 ID 0005-1207-0034-1617 - 06.02.2017
R2_0602 = np.array([[0.5068,   -0.2581,   0.8069],
                  [-0.8471, -0.1547,   0.4827],
                  [-0.0002, 0.9402,    0.3009]])
t2_0602 = np.array([-360.9835, -196.2289,   22.2417])

# PicoFlexx ID 0005-1203-0034-0709 - 29.9
R1_2909 = np.array([[-0.4931184947, -0.8589221835, 0.03191805631], 
                    [-0.3777805865, 0.1835363507, -0.8975127339], 
                    [-0.7720414996, 0.4588017464, 0.4187896848]])
t1_2909 = np.array([362.2033691, -196.6711273, 20.63752747])

# PicoFlexx ID 0005-1203-0034-0515 - 29.9
R2_2909 = np.array([[0.5437250137, -0.8403939009, 0.02766888589], 
                    [-0.3759854436, -0.2135225981, 0.9031654000],
                    [0.7521053553, 0.5008099079, 0.4314989746]])
t2_2909 = np.array([-365.4113770, -200.1092072, 19.15486145])
