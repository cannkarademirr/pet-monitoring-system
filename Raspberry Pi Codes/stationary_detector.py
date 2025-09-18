import numpy as np


stationary_threshold = 15  # piksel cinsinden hareket sınırı
stationary_frame_count = 10  # kaç frame boyunca sabit kalmalı?


def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_stationary(center_history):
    if len(center_history) < stationary_frame_count:
        return False

    # Son N merkez konumuna bakılır
    recent = center_history[-stationary_frame_count:]
    avg_x = sum([pt[0] for pt in recent]) / len(recent)
    avg_y = sum([pt[1] for pt in recent]) / len(recent)

    # Her pozisyonun ortalamaya olan uzaklığı kontrol edilir
    for (x, y) in recent:
        if np.linalg.norm([x - avg_x, y - avg_y]) > stationary_threshold:
            return False
    return True

def on_stationary_object_detected(object_id, class_id, center):
    print(f"[ALERT] Nesne sabit kaldı! ID: {object_id}, Class: {class_id}, Center: {center}")

    # Buraya uyarı gönderme, loglama veya başka bir işlem çağrısı yapılabilir