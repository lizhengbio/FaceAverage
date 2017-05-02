#!/usr/bin/env python

# Copyright (c) 2016 Satya Mallick <spmallick@learnopencv.com>
# All rights reserved. No warranty, explicit or implicit, provided.


import os
import cv2
import numpy as np
import math
import sys

# Read points from text files in directory


def readPoints(path):
    # Create an array of array of points.
    pointsArray = []

    # List all files in the directory and read points from text files one by
    # one
    for filePath in sorted(os.listdir(path)):

        if filePath.endswith(".txt"):

            # Create an array of points.
            points = []

            # Read points from filePath
            with open(os.path.join(path, filePath)) as file:
                for line in file:
                    x, y = line.split()
                    points.append((int(x), int(y)))

            # Store array of points
            pointsArray.append(points)

    return pointsArray

# Read all jpg images in folder.


def readImages(path):

    # Create array of array of images.
    imagesArray = []

    # List all files in the directory and read points from text files one by
    # one
    for filePath in sorted(os.listdir(path)):
        if filePath.endswith(".jpg"):
            # Read image found.
            img = cv2.imread(os.path.join(path, filePath))

            # Convert to floating point
            img = np.float32(img) / 255.0

            # Add to array of images
            imagesArray.append(img)

    return imagesArray

# Compute similarity transform given two sets of two points.
# OpenCV requires 3 pairs of corresponding points.
# We are faking the third one.


def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * \
        (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * \
        (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * \
        (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * \
        (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateRigidTransform(
        np.array([inPts]), np.array([outPts]), False)

    return tform


# Check if a point is inside a rectangle
def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Calculate delanauy triangle


def calculateDelaunayTriangles(rect, points):
    # Create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points:
        subdiv.insert((p[0], p[1]))

    # List of triangles. Each triangle is a list of 3 points ( 6 numbers )
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array

    delaunayTri = []

    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in xrange(0, 3):
                for k in xrange(0, len(points)):
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

    return delaunayTri


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.


def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[
                         1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in xrange(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]                                                          :r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]
        :r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


if __name__ == '__main__':

    path = 'presidents/'

    # Dimensions of output image
    w = 600
    h = 600

    # Read points for all images
    allPoints = readPoints(path)

    # Read all images
    images = readImages(path)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]

    imagesNorm = []
    pointsNorm = []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array([(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2),
                            (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array([(0, 0)] * (len(allPoints[0]) +
                                     len(boundaryPts)), np.float32())

    n = len(allPoints[0])

    numImages = len(images)

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.

    for i in xrange(0, numImages):

        points1 = allPoints[i]

        # Corners of the eye in input image
        eyecornerSrc = [allPoints[i][36], allPoints[i][45]]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        img = cv2.warpAffine(images[i], tform, (w, h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points1), (68, 1, 2))

        points = cv2.transform(points2, tform)

        points = np.float32(np.reshape(points, (68, 2)))

        # Append boundary points. Will be used in Delaunay Triangulation
        points = np.append(points, boundaryPts, axis=0)

        # Calculate location of average landmark points.
        pointsAvg = pointsAvg + points / numImages

        pointsNorm.append(points)
        imagesNorm.append(img)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    # dt = calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Pin down the triangulation mapping to avoid potential failure in Delaunay Triangling.
    dt = [(17, 37, 36), (37, 17, 18), (2, 75, 1), (75, 2, 3), (0, 75, 17), (75, 0, 1), (35, 53, 52), (53, 35, 54), (75, 68, 17), (1, 0, 36), (31, 2, 41), (2, 31, 3), (75, 3, 4), (2, 1, 41), (57, 7, 58), (7, 57, 8), (74, 4, 5), (4, 74, 75), (33, 52, 51), (52, 33, 34), (74, 5, 6), (4, 3, 48), (58, 62, 57), (62, 58, 61), (73, 74, 6), (5, 4, 48), (30, 35, 34), (35, 30, 29), (6, 5, 59), (13, 35, 14), (35, 13, 54), (73, 6, 7), (7, 6, 58), (29, 31, 40), (31, 29, 30), (73, 7, 8), (10, 73, 9), (73, 10, 72), (9, 73, 8), (9, 8, 56), (72, 10, 11), (10, 9, 55), (31, 41, 40), (71, 72, 12), (11, 10, 54), (21, 27, 39), (27, 21, 22), (12, 72, 11), (12, 11, 54), (1, 36, 41), (71, 12, 13), (13, 12, 54), (21, 39, 38), (70, 71, 16), (13, 14, 71), (29, 40, 39), (15, 71, 14), (71, 15, 16), (15, 14, 46), (17, 36, 0), (16, 15, 45), (69, 23, 22), (23, 69, 24), (37, 18, 19), (18, 69, 19), (69, 18, 68), (40, 38, 39), (38, 40, 37), (18, 17, 68), (69, 20, 19), (20, 69, 21), (69, 22, 21),
          (19, 20, 38), (29, 47, 35), (47, 29, 42), (20, 21, 38), (22, 42, 27), (42, 22, 43), (27, 42, 28), (46, 14, 35), (43, 22, 23), (43, 24, 44), (24, 43, 23), (24, 69, 25), (26, 45, 25), (45, 26, 16), (24, 25, 44), (26, 70, 16), (70, 26, 25), (42, 29, 28), (27, 28, 39), (44, 46, 47), (46, 44, 45), (28, 29, 39), (63, 55, 56), (55, 63, 53), (48, 31, 49), (31, 48, 3), (56, 62, 63), (62, 56, 57), (59, 5, 48), (31, 30, 32), (33, 50, 32), (50, 33, 51), (31, 32, 50), (32, 30, 33), (55, 9, 56), (33, 30, 34), (34, 35, 52), (37, 19, 38), (37, 40, 41), (36, 37, 41), (44, 47, 43), (44, 25, 45), (42, 43, 47), (15, 46, 45), (46, 35, 47), (49, 31, 50), (48, 49, 60), (61, 58, 59), (49, 50, 67), (56, 8, 57), (50, 51, 67), (52, 65, 51), (65, 52, 53), (10, 55, 54), (53, 54, 64), (54, 55, 64), (55, 53, 64), (66, 63, 62), (63, 66, 65), (61, 49, 67), (49, 61, 60), (58, 6, 59), (60, 61, 59), (59, 48, 60), (67, 51, 66), (67, 66, 61), (62, 61, 66), (66, 51, 65), (53, 63, 65), (25, 69, 70)]

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in xrange(0, len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in xrange(0, len(dt)):
            tin = []
            tout = []

            for k in xrange(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            warpTriangle(imagesNorm[i], img, tin, tout)

        # Add image intensities for averaging
        output = output + img

    # Divide by numImages to get average
    output = output / numImages

    # Display result
    cv2.imshow('image', output)
    cv2.waitKey(0)
