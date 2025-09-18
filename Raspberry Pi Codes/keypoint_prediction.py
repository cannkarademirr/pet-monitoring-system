import cv2
import numpy as np
import config


def keyPointDetection(frame):

    # Başlangıçta tüm keypoints ve skorlar
    keypoints = {}
    keypoint_scores = {}

    results = config.dogKeyPointDetectionModel(frame, verbose=False)[0]

    for result in results:
        # Keypoint koordinatları
        points = np.array(result.keypoints.xy.cpu(), dtype="int")[0]
        # Skorlar
        scores = result.keypoints.conf.cpu().numpy()[0]

        for i, (p, score) in enumerate(zip(points, scores)):
            if p[0] == 0 and p[1] == 0:
                continue
            name = config.keypoint_names[i] if i < len(config.keypoint_names) else f"kp{i+1}"
            
            # Görselleştirme
            cv2.circle(frame, (p[0], p[1]), 3, (0, 255, 0), -1)
            cv2.putText(frame, f"{name} {score:.2f}", (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            keypoints[name] = [int(p[0]), int(p[1])]
            keypoint_scores[name] = float(score)

    cv2.imwrite("results/keypointDetection.jpg", frame)

    # Hem koordinatlar hem skorlar döndürülür
    return keypoints, keypoint_scores, frame
