"""
Copyright 2015, 2018 ICG, Graz University of Technology

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

import cv2

import numpy


def getTransformationMatrix(center, rot, trans, scale):
    ca = numpy.cos(rot)
    sa = numpy.sin(rot)
    sc = scale
    cx = center[0]
    cy = center[1]
    tx = trans[0]
    ty = trans[1]
    t = numpy.array([ca * sc, -sa * sc, sc * (ca * (-tx - cx) + sa * ( cy + ty)) + cx,
                     sa * sc, ca * sc, sc * (ca * (-ty - cy) + sa * (-tx - cx)) + cy])
    return t


def transformPoint2D(pt, M):
    """
    Transform point in 2D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt2 = numpy.asmatrix(M.reshape((3, 3))) * numpy.matrix([pt[0], pt[1], 1]).T
    return numpy.array([pt2[0] / pt2[2], pt2[1] / pt2[2]])


def transformPoint3D(pt, M):
    """
    Transform point in 3D coordinates
    :param pt: point coordinates
    :param M: transformation matrix
    :return: transformed point
    """
    pt3 = numpy.asmatrix(M.reshape((4, 4))) * numpy.matrix([pt[0], pt[1], pt[2], 1]).T
    return numpy.array([pt3[0] / pt3[3], pt3[1] / pt3[3], pt3[2] / pt3[3]])
    

def pointsImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D
    :param sample: points in (x,y,z) with x,y in image coordinates and z in mm
    z is assumed to be the distance from the camera plane (i.e., not camera center)
    :return: normalized points in mm
    """
    ret = numpy.zeros((sample.shape[0], 3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = pointImgTo3D(sample[i], fx, fy, ux, uy)
    return ret
    

def pointImgTo3D(sample, fx, fy, ux, uy):
    """
    Normalize sample to metric 3D
    :param sample: point in (x,y,z) with x,y in image coordinates and z in mm
    :return: normalized points in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    # convert to metric using f
    ret[0] = (sample[0]-ux)*sample[2]/fx
    ret[1] = (sample[1]-uy)*sample[2]/fy
    ret[2] = sample[2]
    return ret
    

def points3DToImg(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: points in (x,y,z) with x,y and z in mm
    :return: points in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((sample.shape[0], 3), numpy.float32)
    for i in range(sample.shape[0]):
        ret[i] = point3DToImg(sample[i], fx, fy, ux, uy)
    return ret
    

def point3DToImg(sample, fx, fy, ux, uy):
    """
    Denormalize sample from metric 3D to image coordinates
    :param sample: points in (x,y,z) with x,y and z in mm
    :return: points in (x,y,z) with x,y in image coordinates and z in mm
    """
    ret = numpy.zeros((3,), numpy.float32)
    # convert to metric using f
    if sample[2] == 0.:
        ret[0] = ux
        ret[1] = uy
        return ret
    ret[0] = sample[0]/sample[2]*fx+ux
    ret[1] = sample[1]/sample[2]*fy+uy
    ret[2] = sample[2]
    return ret
    
    
def depthimageToPCL(img, fx, fy, ux, uy, maxDepth):
    """
    :param img              depth image
    :param fx, fy, ux, uy   intrinsic camera parameters
    :param maxDepth         IGNORED; 
    # (was: depth pixels above this value are ignored)
    """
    pointcloud = numpy.zeros((img.shape[0] * img.shape[1], 3), dtype=float)
    depth_idx = 0
    for v in range(0, img.shape[0]):
        for u in range(0, img.shape[1]):
            d = img[v,u]
#            # skip invalid points
#            if (d == 0.) or (d > maxDepth):
#                continue
            
            pt3D = pointImgTo3D((u,v,d), fx, fy, ux, uy)

            pointcloud[depth_idx,0] = pt3D[0]
            pointcloud[depth_idx,1] = pt3D[1]
            pointcloud[depth_idx,2] = d

            depth_idx += 1

    return pointcloud[0:depth_idx,:]
    
    
def transform_points_to_other_cam(pts_uvd_cam1, calib_cam1, R_cam1, t_cam1,
                                  calib_cam2, R_cam2, t_cam2):
    """
    assumes distorted points, computes distorted points
    
    Arguments:
        pts_uvd_cam1: Nx3 array, with N 3D points
        calib_cam1/2: list of intrinsic camera parameters:
            fx, fy, cx, cy, k1, k2, p1, p2, k3
            
    Returns:
        distorted uvd points
    """
    # Assemble camera matrices and distortion coefficients
    fx1, fy1, cx1, cy1, k1c1, k2c1, p1c1, p2c1, k3c1 = calib_cam1
    # Cam 1
    K1 = numpy.eye(3)
    K1[0,0] = fx1
    K1[1,1] = fy1
    K1[0,2] = cx1
    K1[1,2] = cy1
    distortion1 = numpy.array([k1c1, k2c1, p1c1, p2c1, k3c1])
    # Cam 2
    fx2, fy2, cx2, cy2, k1c2, k2c2, p1c2, p2c2, k3c2 = calib_cam2
    K2 = numpy.eye(3)
    K2[0,0] = fx2
    K2[1,1] = fy2
    K2[0,2] = cx2
    K2[1,2] = cy2
    distortion2 = numpy.array([k1c2, k2c2, p1c2, p2c2, k3c2])
    # Compute inverse Rotation
    R2inv = numpy.linalg.inv(R_cam2)
        
    # Undistort points
    gtImg2d = pts_uvd_cam1[:,0:2]
    gtImg2d = numpy.expand_dims(gtImg2d, axis=0)
    gtImg2dUndist = cv2.undistortPoints(gtImg2d, K1, distortion1)
    gtImg2dUndist = numpy.squeeze(gtImg2dUndist)
    # Project points to 3D
    gt3dUndist = gtImg2dUndist * numpy.expand_dims(pts_uvd_cam1[:,2], axis=1)    # coords from cv2.undistortPoints are already normalized, i.e., unit-converted on camera plane. Hence we just multiply x and y with z
    gt3dUndist = numpy.append(gt3dUndist, numpy.expand_dims(pts_uvd_cam1[:,2], axis=1), axis=1)
    # Transform points to cam2 coords
    gt3dCam2Undist = (R2inv.dot(R_cam1.dot(gt3dUndist.T) + numpy.expand_dims(t_cam1 - t_cam2, axis=1))).T
    # Project points to image coords and distort it
    gtImg2d, _ = cv2.projectPoints(gt3dCam2Undist, numpy.zeros((3,1)), numpy.zeros((3,1)), K2, distortion2)
    gtImg2d = numpy.squeeze(gtImg2d)
    pts_uvd_cam2 = numpy.append(gtImg2d, numpy.expand_dims(gt3dCam2Undist[:,2], axis=1), axis=1)
#    # Project distorted points to 3D
#    pts_3d_cam2 = pointsImgTo3D(pts_uvd_cam2, fx2, fy2, cx2, cy2)
    
    return pts_uvd_cam2
    