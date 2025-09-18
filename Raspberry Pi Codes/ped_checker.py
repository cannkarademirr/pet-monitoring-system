import cv2
import numpy as np

def is_point_in_polygon_with_tolerance(point, polygon, tolerance=5):
    """
    Noktanın poligon içinde olup olmadığını kontrol eder, belirli bir toleransla.
    """
    distance = cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, True)
    return distance >= -tolerance

def isObject_onThePAD(keypoints, pad_polygon, tolerance=5):
    """
    Köpeğin patilerinden herhangi biri pad poligonun içinde mi?
    
    keypoints: {'Front Left Paw': (x, y), ...}
    pad_polygon: [(x1, y1), (x2, y2), ...]
    tolerance: poligon dışında kabul edilecek maksimum mesafe (piksel)
    
    Dönüş: True veya False
    """
    paw_names = [
        "Front Left Paw", "Front Right Paw",
        "Rear Left Paw", "Rear Right Paw"
    ]
    
    for paw in paw_names:
        if paw in keypoints:
            point = tuple(keypoints[paw])
            if is_point_in_polygon_with_tolerance(point, pad_polygon, tolerance):
                return True  # En az bir pati pad üstünde
    
    return False  # Hiçbiri pad üstünde değil
