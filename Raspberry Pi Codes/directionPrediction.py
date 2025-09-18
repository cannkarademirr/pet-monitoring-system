
def get_vector(a, b):
    """İki nokta arasındaki yön vektörü hesaplanır."""
    start_point = (int(a[0]), int(a[1]))
    end_point = (int(b[0]), int(b[1]))

    return (b[0] - a[0], b[1] - a[1]), start_point, end_point



def compute_candidate_vector(point_keys_front, point_keys_back, keypoints):
    """
    Belirtilen ön ve arka noktalar grubundan bir vektör hesaplar.
    Dönen tuple: (vektör, toplam kullanılan nokta sayısı)
    """
    front_pts = [keypoints.get(k) for k in point_keys_front]
    back_pts = [keypoints.get(k) for k in point_keys_back]
    
    front_avg, count_front = average_points(front_pts)
    back_avg, count_back = average_points(back_pts)
    
    if front_avg is None or back_avg is None:
        return None, 0
    
    vec = get_vector(back_avg, front_avg)
    weight = count_front + count_back  # Ne kadar nokta kullanıldıysa, o kadar güvenilir kabul edelim.
    return vec, weight

def combine_vectors(candidate_vectors):
    """
    Farklı candidate vektörlerini ağırlıklı olarak birleştirir.
    Eğer hiç candidate bulunamazsa None döner.
    """
    total_weight = 0
    sum_x, sum_y = 0, 0
    for vec, weight in candidate_vectors:
        if vec is None or weight == 0:
            continue
        sum_x += vec[0] * weight
        sum_y += vec[1] * weight
    if total_weight == 0:
        return None
    return (sum_x / total_weight, sum_y / total_weight)

#yön vektöürünü çizmek için eklendi
def get_line_points(p1, p2):
    """
    p1 ve p2, (x, y) tuple'ları olarak verilen iki nokta.
    Bu fonksiyon, p1'den p2'ye giden doğru üzerindeki tüm piksel koordinatlarını 
    Bresenham algoritmasını kullanarak döndürür.
    """
    x1, y1 = p1
    x2, y2 = p2
    
    points = []
    
    # Farkları ve adım yönlerini hesapla
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    
    # İlk hata değeri
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if (x1, y1) == (x2, y2):
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

def compute_y_for_x(sp, ep, x_value):
    """
    sp: (x1, y1) başlangıç noktası
    ep: (x2, y2) bitiş noktası
    x_value: y'sini hesaplamak istediğin x değeri
    """
    x1, y1 = sp
    x2, y2 = ep
    # x2 ile x1 eşit ise, doğru dikeydir ve bu durumda y değeri sabit değildir.
    if x2 == x1:
        x2 += 0.1
        
    # Eğimi hesapla
    m = (y2 - y1) / (x2 - x1)
    # y-kesişimini hesapla: b = y1 - m * x1
    b = y1 - m * x1
    # Belirtilen x için y değeri
    y = m * x_value + b
    return int(y)

def infer_dog_direction(keypoints: dict, frame): #frame yön vektöürünün çizilmesi için eklenmiştir kaldırulacak 
    """
    Köpeğin yönünü, elimizdeki keypointlerden tahmin eder.
    Bu versiyonda hem 'direction' (string: 'up', 'down', 'left', 'right' veya 'unknown') 
    hem de açı (angle, derece cinsinden) döndürülüyor.
    
    Öncelik: eğer "Nose", "Chin" gibi kafa bilgileri varsa bunlarla;  
    eğer yoksa vücut (ön vs. arka nokta) bilgileri kullanılır.
    """
    # Kafa grubundan (örn: Nose & Chin) ortalama nokta
    head, n_front = average_points([
        keypoints.get("Nose"), keypoints.get("Chin"),
        keypoints.get("Left Ear Base"), keypoints.get("Right Ear Base"),
        keypoints.get("Left Ear Tip"), keypoints.get("Right Ear Tip")
        ])
    
    # Kulaklardan destek alınabilir:
    back, n_back = average_points([
        keypoints.get("Tail Start"), keypoints.get("Tail End")
    ])
    
    vector_used = None
    if head is not None and back is not None:
        back = (back[0], back[1] - 60) ## to normalize
        vector_used, sp, ep = get_vector(back, head)
    else:
        # Vücut bilgisi: Ön ve arka noktaların ortalaması
        front, n_front_body = average_points([
            keypoints.get("Front Left Knee"), keypoints.get("Front Right Knee"),
            keypoints.get("Front Left Elbow"), keypoints.get("Front Right Elbow"),
            keypoints.get("Nose"), keypoints.get("Chin"),
            keypoints.get("Left Ear Base"), keypoints.get("Right Ear Base"),
            keypoints.get("Left Ear Tip"), keypoints.get("Right Ear Tip")
        ])
        
        rear, n_back_body = average_points([
            keypoints.get("Rear Left Knee"), keypoints.get("Rear Right Knee"),
            keypoints.get("Rear Left Elbow"), keypoints.get("Rear Right Elbow"),
            keypoints.get("Tail Start"), keypoints.get("Tail End")
        ])

        if front is not None and rear is not None:
            rear = (rear[0], rear[1] - 50) ## to normalize
            vector_used, sp, ep = get_vector(rear, front)
        elif front is None and rear is not None and head is not None:
            avrg_knee = average_points([
                keypoints.get("Rear Left Knee"), keypoints.get("Rear Right Knee"),
            ])
            avrg_elbow = average_points([
                keypoints.get("Rear Left Elbow"), keypoints.get("Rear Right Elbow"),
            ])
            if avrg_knee is not None and avrg_elbow is not None: 
                head = (head[0], head[1] - abs(avrg_knee[1]- avrg_elbow[1]))
                vector_used, sp, ep = get_vector(rear, head)
        elif rear is None and front is not None and back is not None:
            avrg_knee = average_points([
                keypoints.get("Front Left Knee"), keypoints.get("Front Right Knee"),
            ])
            avrg_elbow = average_points([
                keypoints.get("Front Left Elbow"), keypoints.get("Front Right Elbow"),
            ])
            if avrg_knee is not None and avrg_elbow is not None: 
                back = (head[0], back[1] - abs(avrg_knee[1]- avrg_elbow[1]))
                vector_used, sp, ep = get_vector(back, front)
        else:
            if keypoints.get('Chin') and keypoints.get('Nose') and keypoints.get("Left Ear Base") and keypoints.get("Right Ear Base") and keypoints.get("Left Ear Tip") and keypoints.get("Right Ear Tip"):
                vector_a = (1, 0)    # X ekseni yönünde (sağa doğru)
                vector_b = (0, -1)   # Y ekseni yönünde (aşağıya doğru)
                vector_used, sp, ep = get_vector(vector_a, vector_b)
            elif keypoints.get('Tail Start') and keypoints.get('Tail End') and keypoints.get('Withers'):
                vector_a = (1, 0)    # X ekseni yönünde (sağa doğru)
                vector_b = (0, 1)    # Y ekseni yönünde (yukarıya doğru)
                vector_used, sp, ep = get_vector(vector_a, vector_b)
        
        
    if vector_used is None:
        return "unknown", None , frame
    
    '''
    Yön vektöürünü ekrana çiz
    import cv2 ## yön vektröü çizilmesi için eklendi kalkacak
    '''
    import cv2
    line_points = get_line_points(sp, ep)
    # x = 0 için y
    y_at_0 = compute_y_for_x(sp, ep, 0)
    sp = (0, y_at_0)
# x = 640 için y
    y_at_640 = compute_y_for_x(sp, ep, 640)
    ep = (640, y_at_640)
    cv2.arrowedLine(frame, sp, ep, color=(255, 0, 0), thickness=1, tipLength=0.2)
    dx, dy = vector_used
    # atan2 kullanarak açıyı hesaplıyoruz (ekran koordinatında y yukarı negatif)
    angle = math.degrees(math.atan2(-dy, dx))
    cv2.imwrite("results/keypointDetection.jpg", frame)
    
    # Yön sınıflandırması
    if -45 <= angle <= 45:
        direction = "right"
    elif 45 < angle <= 135:
        direction = "up"
    elif angle > 135 or angle < -135:
        direction = "left"
    elif -135 <= angle < -45:
        direction = "down"
    else:
        direction = "unknown"
    
    return direction, angle, frame

import math

def average_points(points):
    """Verilen noktalardan (None olmayanlar) ortalama noktayı ve kullanılan nokta sayısını döner."""
    valid = [p for p in points if p is not None]
    if not valid:
        return None, 0
    avg_x = sum(p[0] for p in valid) / len(valid)
    avg_y = sum(p[1] for p in valid) / len(valid)
    return (avg_x, avg_y), len(valid)


