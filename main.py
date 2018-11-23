#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from imutils.face_utils import rect_to_bb
import dlib
import imutils
import os, time
import os.path
import numpy as np

#-------------------------------------------

mediaType = "video"  # image / video
imageFolder = "/media/sf_VMshare/pics"
videoFile = "/media/sf_VMshare/VIRB0047.MP4"
videoOutFile = "/media/sf_VMshare/test_out.avi"

datasetPath = "eyeAutoLabled/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

labelName = "eyes"
minEyeSize = (10, 10)
maxImageWidth = 2000

landmarksDB = "dlib/shape_predictor_68_face_landmarks.dat"
dlib_detectorRatio = 2
folderCharacter = "/"  # \\ is for windows
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

def chkEnv():
    if not os.path.exists(landmarksDB):
        print("There is no landmark db file for this path:", landmarksDB)
        quit()

    if(mediaType=="image" and (not os.path.exists(imageFolder))):
        print("There is no folder for this path:", imageFolder)
        quit()

    if(mediaType=="video" and (not os.path.exists(videoFile))):
        print("There is no video file for this path:", videoFile)
        quit()

    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    if not os.path.exists(datasetPath + imgPath):
        os.makedirs(datasetPath + imgPath)

    if not os.path.exists(datasetPath + labelPath):
        os.makedirs(datasetPath + labelPath)

def getEyeShapes(landmarks):
    #right eye: 36~41
    #left eye: 42~47
    eyes = []

    for id in range(36,42):
        eyes.append((landmarks.part(id).x, landmarks.part(id).y))

    for id in range(42,48):
        eyes.append((landmarks.part(id).x, landmarks.part(id).y))

    eyes_np = np.array(eyes)
    bbox = cv2.boundingRect(eyes_np)

    return bbox

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(img, filename, fullpath, bboxes):
    xmlObject = ""
    for bbox in bboxes:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(img, bboxes):
    filename = str(time.time())
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".xml"

    cv2.imwrite(datasetPath + imgPath + jpgFilename, img)

    xmlContent = generateXML(img, xmlFilename, datasetPath + labelPath + xmlFilename, bboxes)
    file = open(datasetPath + labelPath + xmlFilename, "w")
    file.write(xmlContent)
    file.close

def labelEyes(img):
    detector = dlib.get_frontal_face_detector()

    if(img.shape[1]>maxImageWidth):
        img = imutils.resize(img, width=maxImageWidth)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    eyesBBOX = []
    for faceid, rect in enumerate(rects):
        shape = predictor(gray, rect)
        eyes_bbox = getEyeShapes(shape)
        eyesBBOX.append(eyes_bbox)

    makeLabelFile(img, eyesBBOX)

    return eyesBBOX

#--------------------------------------------

chkEnv()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarksDB)

if(mediaType=="image"):
    for file in os.listdir(imageFolder):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            print("Processing: ", imageFolder + folderCharacter + file)

            image = cv2.imread(imageFolder + folderCharacter + file)
            labelEyes(image)

elif(mediaType=="video"):
    camera = cv2.VideoCapture(videoFile)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    if(videoOutFile != ""):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(videoOutFile, fourcc, 30.0, (int(width),int(height)))

    grabbed = True

    i = 0
    while grabbed:
        i += 1
        (grabbed, frame) = camera.read()
        eyes = labelEyes(frame)

        for eye in eyes:
            cv2.rectangle( frame,(eye[0],eye[1]),(eye[0]+eye[2],eye[1]+eye[3]),(0,255,0),2)

        if(videoOutFile != ""):
            out.write(frame)

        cv2.imshow("FRAME", imutils.resize(frame, width=640))
        print("Frame #{}".format(i))

        cv2.waitKey(1)
