
import cv2
from ultralytics import YOLO


font = cv2.FONT_HERSHEY_SIMPLEX
text_color_b = (0,0,0) # black
text_color_w = (255,255,255) # white

confidence_score = 0.5
sync_time = 100000000000000000000000

pooping_treshold = 100

recorded_video_seconds = 60
recorded_video_frames = 30
###### - MODELS - ######

dogKeyPointDetectionModel = YOLO("models/YoloV11n-dogKeypointDetection.pt")
pooping_detection = YOLO("models/pooping_detection.pt")

keypoint_names = [
    "Front Left Paw",   # 0
    "Front Left Knee",  # 1
    "Front Left Elbow", # 2
    "Rear Left Paw",    # 3
    "Rear Left Knee",   # 4
    "Rear Left Elbow",  # 5
    "Front Right Paw",  # 6
    "Front Right Knee", # 7
    "Front Right Elbow",# 8
    "Rear Right Paw",   # 9
    "Rear Right Knee",  #10
    "Rear Right Elbow", #11
    "Tail Start",       #12
    "Tail End",         #13
    "Left Ear Base",    #14
    "Right Ear Base",   #15
    "Nose",             #16
    "Chin",             #17
    "Left Ear Tip",     #18
    "Right Ear Tip",    #19
    "Left Eye",         #20
    "Right Eye",        #21
    "Withers",          #22
    "Throat"            #23
]





