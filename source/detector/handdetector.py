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

import math
import numpy
import cv2
from scipy import ndimage

        
class HandDetectorICG(object):
    """
    Detect hand based on simple heuristics (closest object to ref. plane, ...)
    """
    
    BACKGROUND_VALUE = 10000
    
    RESIZE_CV2_NN = 1
    RESIZE_CV2_LINEAR = 2

    def __init__(self):
        """
        Constructor
        """
        self.binSizeDepthHistogram = 20      # bin size of histogram to find closest point (in mm)
        self.minNumObjectPointsInBin = 30    # number of points, which should be inside a bin of the depth histogram, this is to get rid of noise
        self.minGrayValueDetection = 50      # minimum value of IR/gray for a valid pixel for detection (we can use a stricter threshold here)
        # depth resize method
        self.resizeMethod = self.RESIZE_CV2_NN


    def calculateCoM(self, dpt):
        """
        Calculate the center of mass
        :param dpt: depth image; invalid pixels which should not be considered must be set zero
        :param minDepth: minimum depth threshold; pixels with lower depth are not considered
        :param maxDepth: maximum depth threshold; pixels with higher depth are not considered
        :return: (x,y,z) center of mass
        """
        dc = dpt.copy()
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = numpy.count_nonzero(dc)
        com = numpy.array((cc[1]*num, cc[0]*num, dc.sum()), numpy.float)

        if num == 0:
            return numpy.array((0, 0, 0), numpy.float)
        else:
            return com/num


    def transformPCLWithConstraints(self, pcl,
                                                  numBins,
                                                  R, t, 
                                                  minD, maxD,
                                                  minX, maxX, minY, maxY, minZ, maxZ):
        """
        Transform the point cloud by considering some constraints, 
        i.e., the resulting PCL only contains points fullfiling the constraints
        :param pcl          the point cloud
        :param numBins      number of histogram bins for returned histogram of depth values in the transformed space
        :param R            rotation matrix
        :param t            translation vector
        :param minD/maxD    minimum/maximum depth/z-value in original space
        """
        tColumn = t.copy()
        tColumn.shape = (3,1)
        pclOut = (R.dot(pcl.T) + tColumn).T
        
        # Apply constraints in interaction space
        pclOut = pclOut[pclOut[:,0] >= minX,:]
        pclOut = pclOut[pclOut[:,0] <= maxX,:]
        pclOut = pclOut[pclOut[:,1] >= minY,:]
        pclOut = pclOut[pclOut[:,1] <= maxY,:]
        pclOut = pclOut[pclOut[:,2] >= minZ,:]
        pclOut = pclOut[pclOut[:,2] <= maxZ,:]
        
        # Update depth histogram
        depthHistogram = numpy.histogram(pclOut[:,2], bins=numBins, range=(0,maxZ), density=False)[0]
        
        return pclOut, depthHistogram
        
        
    def splitPCLIntoSlices(self, pcl, maxZ):
        """
        Split the given pcl into slices according to the depth bins of the depth-histogram
        """
        binEdges = range(0,int(maxZ),self.binSizeDepthHistogram)
        pclSliced = [pcl[numpy.bitwise_and(pcl[:,2] >= i, pcl[:,2] < (i+self.binSizeDepthHistogram)),:] for i in binEdges]
        
        return pclSliced
        
        
    def refineObjectComInOrthoFromPoint_Markus(self, 
                                               imgDepth, 
                                               startPointUvd, 
                                               handBBSizePx, 
                                               handBBSizeMM,
                                               maxNumIterations=4):
        """
        Finds the center of mass of an object in an ortho-map 
        given a masked region on the object (usually the closest region)
        (This is derived from the original code from Markus - using numpy and
        OpenCV functions)
        :param imgDepth         depth image; orthogonal projection
        :param imgMaskStartROI  binary mask specifying the start region 
                                (may contain more than one foreground region)
        :param handBBSizePx     the hand bounding box side length in pixels
                                (since we work on an ortho-map the size 
                                in image coordinates is independent of the depth)
        :param handBBSizeMM     the hand bounding box side length in millimeters
        :return center-of-mass of the object in uvd, i.e., x,y in pixels and depth in millimeters
        """
        handBBHalfsizePx = handBBSizePx / 2.
        handBBHalfsizePxInt = int(round(handBBSizePx / 2.))
        handBBHalfsizeMM = handBBSizeMM / 2.
        
        cx = int(round(startPointUvd[0]))
        cy = int(round(startPointUvd[1]))
        cz = startPointUvd[2]

        # crop
        xstart = int(max(   cx-handBBHalfsizePxInt, 0))
        xend = int(min(     cx+handBBHalfsizePxInt, imgDepth.shape[1]))
        ystart = int(max(   cy-handBBHalfsizePxInt, 0))
        yend = int(min(     cy+handBBHalfsizePxInt, imgDepth.shape[0]))

        cropped = imgDepth[ystart:yend, xstart:xend].copy()
        cropped[cropped < cz-handBBHalfsizeMM] = 0.
        cropped[cropped > cz+handBBHalfsizeMM] = 0.
        com = self.calculateCoM(cropped)
        if numpy.allclose(com, 0.):
            com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
        com[0] += xstart
        com[1] += ystart

        # refine iteratively
        for k in range(maxNumIterations):
            # calculate boundaries
            zstart = com[2] - handBBHalfsizeMM
            zend = com[2] + handBBHalfsizeMM
            xstart = int(max(round( com[0] - handBBHalfsizePx), 0.))
            xend = int(min(round(   com[0] + handBBHalfsizePx), imgDepth.shape[1]))
            ystart = int(max(round( com[1] - handBBHalfsizePx), 0.))
            yend = int(min(round(   com[1] + handBBHalfsizePx), imgDepth.shape[0]))

            # crop
            cropped = imgDepth[ystart:yend, xstart:xend].copy()
            cropped[cropped < zstart] = 0.
            cropped[cropped > zend] = 0.

            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
            com[0] += xstart
            com[1] += ystart

        # final result
        return com
        
        
    def createOrthoDepthMap(self, pcl, minX, maxX, minY):
        """
        Create orthogonal projection to 2D image, where each pixel encodes the depth, 
        i.e., the distance of the projection along z
        :param pcl          point cloud to be projected
        :param minX, maxX,...   xyz limits of the pcl coordinate space
        :return imgOrtho    ortho-depth-map
                s           millimeter to pixel scale
        """        
        # Orthogonal projection along z-axis in reference space
        # Image size
        w = 200
        h = 200
        
        imgOrtho = numpy.zeros((w,h))
        s = float(w) / (maxX - minX)
        for i in range(0,len(pcl)):
            x = int(round((pcl[i,0] - minX) * s))
            y = int(round((pcl[i,1] - minY) * s))
            if ((x >= 0) and (x < w) and (y >= 0) and (y < h)):
                imgOrtho[y,x] = pcl[i,2]
                
        return imgOrtho, s
        
        
    def findPointWithNNeighbors(self, pcl, n, radius):
        """
        Find a point with enough neighbors
        
        :param pcl
        :param n        minimum number of neighbors
        :param radius   maximum distance for being a neighbor
        :return index of point in pcl
                -1 if no point with enough neighbors were found
        """
        for i in range(0,pcl.shape[0]):
            numNeighbors = 0
            for j in range(0,pcl.shape[0]):
                if i is not j:
                    d = numpy.linalg.norm(pcl[i,:] - pcl[j,:])
                    if d < radius:
                        numNeighbors += 1
                    if numNeighbors >= n:
                        return i
                        
        return -1
        
        
    def detectSingleObjectInPCLSliceAndOrthomap(self, pcl, pclSliceClosestPoint, handDiameterMM,
                                                 minX, maxX, minY):
        """
        Detect (single) closest object in ortho-map; 
        start search from point found in slice of pcl with closest point to screen
        :return center of mass in 3D coordinates
        """
        MAX_NUM_ITER = 0
        MIN_NUM_NEIGHBORS = 4
        NEIGHBORHOOD_RADIUS = 8     # in mm
        
        ind = self.findPointWithNNeighbors(pclSliceClosestPoint, MIN_NUM_NEIGHBORS, NEIGHBORHOOD_RADIUS)
        if ind < 0:
            print("No point with enough neighbors found. min. {} neighbors required within {}mm".format(MIN_NUM_NEIGHBORS, NEIGHBORHOOD_RADIUS))
            # No point with enough neighbors found => only noise!?
            return numpy.zeros(3)
        
        imgDepth, mm2px = self.createOrthoDepthMap( pcl, minX, maxX, minY )
                
        # Refine CoM
        closestPointOnObject = numpy.zeros(3)
        closestPointOnObject[0] = (pclSliceClosestPoint[ind,0] - minX) * mm2px
        closestPointOnObject[1] = (pclSliceClosestPoint[ind,1] - minY) * mm2px
        closestPointOnObject[2] = pclSliceClosestPoint[ind,2]
        com = self.refineObjectComInOrthoFromPoint_Markus(imgDepth, closestPointOnObject, 
                                                          (handDiameterMM * mm2px), handDiameterMM, 
                                                          maxNumIterations=MAX_NUM_ITER)
        
        # uvd to xyz; Transform pixel coords to 3D coords
        com[0] = (com[0] / mm2px) + minX
        com[1] = (com[1] / mm2px) + minY
        return com
        
        
    def detectFromPCLs(self, pclCam1, pclCam2,
                       gry4PclCam1, gry4PclCam2,
                       R1, t1, R2, t2,  
                       minX, maxX, minY, maxY, minZ, maxZ,
                       handcubesize,
                       minDepthCam=0, maxDepthCam=9999):
        """
        Detect the closest object (hand) given the PCLs from both cameras
        
        :return     Object center location for both cameras 
                    in camera centered xyz coordinates; 
                    two zero vectors if no object was detected
        """
        # Transform to joint-/reference-space (for detection) (done in loop for easy transfer to C-code and checking thresholds)
        numBins = int(numpy.ceil(maxZ / self.binSizeDepthHistogram))
        pclRef = None
        depthHistogram = numpy.zeros( numBins, dtype=numpy.int )
        depthHistogram_C1 = numpy.zeros( numBins, dtype=numpy.int )
        depthHistogram_C2 = numpy.zeros( numBins, dtype=numpy.int )
        # Cam1
        # Apply constraints on input (depth range, gray range)
        gry4PclCam1 = gry4PclCam1[  pclCam1[:,2] < maxDepthCam]  # keep gry4Pcl aligned with pcl
        pclCam1     = pclCam1[      pclCam1[:,2] < maxDepthCam,:]
        gry4PclCam1 = gry4PclCam1[  pclCam1[:,2] > minDepthCam]  # keep gry4Pcl aligned with pcl
        pclCam1     = pclCam1[      pclCam1[:,2] > minDepthCam,:]
        pclCam1     = pclCam1[      gry4PclCam1 >= self.minGrayValueDetection,:]
        gry4PclCam1 = gry4PclCam1[  gry4PclCam1 >= self.minGrayValueDetection]  # keep gry4Pcl aligned with pcl
    
        pclRef, depthHistogram_C1 = self.transformPCLWithConstraints(pclCam1,  
                                                       numBins,
                                                       R1, t1,
                                                       minDepthCam, maxDepthCam, 
                                                       minX, maxX, minY, maxY, minZ, maxZ)
                                                       
        # Cam2
        # Apply constraints on input (depth range, gray range)
        gry4PclCam2 = gry4PclCam2[  pclCam2[:,2] < maxDepthCam]  # keep gry4Pcl aligned with pcl
        pclCam2     = pclCam2[      pclCam2[:,2] < maxDepthCam,:]
        gry4PclCam2 = gry4PclCam2[  pclCam2[:,2] > minDepthCam]  # keep gry4Pcl aligned with pcl
        pclCam2     = pclCam2[      pclCam2[:,2] > minDepthCam,:]
        pclCam2     = pclCam2[      gry4PclCam2 >= self.minGrayValueDetection,:]
        gry4PclCam2 = gry4PclCam2[  gry4PclCam2 >= self.minGrayValueDetection]  # keep gry4Pcl aligned with pcl
        
        pclRefCam2, depthHistogram_C2 = self.transformPCLWithConstraints(pclCam2,  
                                                       numBins,
                                                       R2, t2,
                                                       minDepthCam, maxDepthCam,
                                                       minX, maxX, minY, maxY, minZ, maxZ)
        pclRef = numpy.append(pclRef, pclRefCam2, axis=0)
                
        pclRefSlices = self.splitPCLIntoSlices(pclRef, maxZ)
        
        depthHistogram = depthHistogram_C1 + depthHistogram_C2
                            
        # Find distance of closest point to screen
        i_min = -1
        for i in range(0,len(depthHistogram)):
            if depthHistogram[i] > self.minNumObjectPointsInBin:
                i_min = i
                break
        
        if i_min == -1:
            return numpy.zeros(3), numpy.zeros(3)
            
        # Detect closest object
        handDiameterMM = handcubesize[0]
        com = self.detectSingleObjectInPCLSliceAndOrthomap(pclRef, pclRefSlices[i_min], handDiameterMM, 
                                                      minX, maxX, minY)
                                                      
        if numpy.allclose(com[2], 0.):
            return numpy.zeros(3), numpy.zeros(3)
        else:
            # Transform com to camera coords
            comCam1 = numpy.linalg.inv(R1).dot(com - t1)
            comCam2 = numpy.linalg.inv(R2).dot(com - t2)
            return comCam1, comCam2
            

    def cropArea3D(self, imgDepth, com, fx, fy, minRatioInside=0.75, 
                   size=(250, 250, 250), dsize=(128, 128), docom=False):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        CROP_BG_VALUE = 0.0

        if len(size) != 3 or len(dsize) != 2:
            raise ValueError("Size must be 3D and dsize 2D bounding box")

        # calculate boundaries
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(math.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
        xend = int(math.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
        ystart = int(math.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
        yend = int(math.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))
        
        # Check if part within image is large enough; otherwise stop
        xstartin = max(xstart,0)
        xendin = min(xend, imgDepth.shape[1])
        ystartin = max(ystart,0)
        yendin = min(yend, imgDepth.shape[0])        
        ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
        if (ratioInside < minRatioInside) and ((com[0] < 0) or (com[0] >= imgDepth.shape[1]) or (com[1] < 0) or (com[1] >= imgDepth.shape[0])):
            print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
            raise UserWarning('Hand not inside image')

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]), max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, imgDepth.shape[0])), (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, imgDepth.shape[1]))), mode='constant', constant_values=int(CROP_BG_VALUE))
        msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
        cropped[msk1] = CROP_BG_VALUE    # backface is at 0, it is set later; setting anything outside cube to same value now (was set to zstart earlier)
        cropped[msk2] = CROP_BG_VALUE    # backface is at 0, it is set later

        # for simulating COM within cube
        if docom is True:
            com = self.calculateCoM(cropped)
            if numpy.allclose(com, 0.):
                com[2] = cropped[cropped.shape[0]//2, cropped.shape[1]//2]
            com[0] += xstart
            com[1] += ystart

            # calculate boundaries
            zstart = com[2] - size[2] / 2.
            zend = com[2] + size[2] / 2.
            xstart = int(math.floor((com[0] * com[2] / fx - size[0] / 2.) / com[2]*fx))
            xend = int(math.floor((com[0] * com[2] / fx + size[0] / 2.) / com[2]*fx))
            ystart = int(math.floor((com[1] * com[2] / fy - size[1] / 2.) / com[2]*fy))
            yend = int(math.floor((com[1] * com[2] / fy + size[1] / 2.) / com[2]*fy))

            # crop patch from source
            cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]), max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
            # add pixels that are out of the image in order to keep aspect ratio
            cropped = numpy.pad(cropped, ((abs(ystart)-max(ystart, 0), abs(yend)-min(yend, imgDepth.shape[0])), (abs(xstart)-max(xstart, 0), abs(xend)-min(xend, imgDepth.shape[1]))), mode='constant', constant_values=0)
            msk1 = numpy.bitwise_and(cropped < zstart, cropped != 0)
            msk2 = numpy.bitwise_and(cropped > zend, cropped != 0)
            cropped[msk1] = zstart
            cropped[msk2] = 0.  # backface is at 0, it is set later

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = numpy.asmatrix(numpy.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        # Compute size of image patch for isotropic scaling where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
        if wb > hb:
            sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
        else:
            sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

        # Compute scale factor from cropped ROI in image to fixed size image patch; set up matrix with same scale in x and y (preserving aspect ratio)
        roi = cropped
        if roi.shape[0] > roi.shape[1]: # Note, roi.shape is (y,x) and sz is (x,y)
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
        else:
            scale = numpy.asmatrix(numpy.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
        scale[2, 2] = 1

        # depth resize
        if self.resizeMethod == self.RESIZE_CV2_NN:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        elif self.resizeMethod == self.RESIZE_CV2_LINEAR:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_LINEAR)
        else:
            raise NotImplementedError("Unknown resize method!")

        # Sanity check
        numValidPixels = numpy.sum(rz != CROP_BG_VALUE)
        if (numValidPixels < 40) or (numValidPixels < (numpy.prod(dsize) * 0.01)):
            print("Too small number of foreground/hand pixels (={})".format(numValidPixels))
            raise UserWarning("No valid hand. Foreground region too small.")

        # Place the resized patch (with preserved aspect ratio) in the center of a fixed size patch (padded with default background values)
        ret = numpy.ones(dsize, numpy.float32) * CROP_BG_VALUE  # use background as filler
        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        # print rz.shape
        off = numpy.asmatrix(numpy.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        # Transformation from original image to fixed size crop includes 
        # the translation of the "anchor" point of the crop to origin (=trans), 
        # the (isotropic) scale factor (=scale), and
        # the offset of the patch (with preserved aspect ratio) within the fixed size patch (=off)
        return ret, off * scale * trans, com
        