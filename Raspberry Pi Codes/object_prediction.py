
import config
import cv2
import numpy as np
def objectDetection(frame):
    score = None
    class_id = None
    results = config.pooping_detection(frame, verbose=False)[0]
    boxes = np.array(results.boxes.data.tolist())
    for box in boxes:
    # print("[INFO].. Box:", box)
        x1, y1, x2, y2, score, class_id = box
        x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
        # print("[INFO].. Box:", x1, y1, x2, y2)
        # print("[INFO].. Class:", class_id)
        # print("[INFO].. Score:", score)
        color = (255, 0, 0)

        
        if score > config.confidence_score:
        
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            score = score * 100
            class_name = results.names[class_id]
            text = f"{class_name}: %{score:.2f}"
            text_loc = (x1, y1-10)
            labelSize, baseLine = cv2.getTextSize(text, config.font, 1, 1)
            cv2.rectangle(frame, 
                        (x1, y1 - 10 - labelSize[1]), 
                        (x1 + labelSize[0], int(y1 + baseLine-10)), 
                        color, 
                        cv2.FILLED)
            cv2.putText(frame, text, (x1, y1-10), config.font, 1, config.text_color_w, thickness=1)
            cv2.imwrite("results/Object_Detection.jpg", frame)
        
        if class_id == 1:
            color = (0, 0, 255)
            return score, True

    return None, False