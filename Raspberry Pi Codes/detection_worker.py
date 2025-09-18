# detection_worker.py
from object_prediction import objectDetection

def detection_worker(frame, result_queue):
    result = objectDetection(frame)
    result_queue.put(result)
