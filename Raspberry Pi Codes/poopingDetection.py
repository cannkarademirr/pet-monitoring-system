import math
import queue
import threading

from sqlalchemy import result_tuple
import keypoint_prediction
import directionPrediction
import config
from colorama import init, Fore, Style

from detection_worker import detection_worker
from servo_control import set_angle




def pad_check_thread(keypoints, pad_coordinates, result_queue, tolerance=5):
    from ped_checker import isObject_onThePAD
    result = isObject_onThePAD(keypoints, pad_coordinates, tolerance)
    result_queue.put(result)

def compute_distance(p1, p2):
    """
    İki nokta (x1, y1) ve (x2, y2) arasındaki Öklit mesafesini hesaplar.
    p1: (x1, y1)
    p2: (x2, y2)
    return: float
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def average_points(points):
    """
    Verilen nokta listesinden (None olmayanları) ortalama konum hesaplanır.
    Ek olarak kullanılan nokta sayısı (güvenirlik) de döndürülür.
    """
    valid = [p for p in points if p is not None]
    if not valid:
        return None, 0
    x = sum(p[0] for p in valid) / len(valid)
    y = sum(p[1] for p in valid) / len(valid)
    return (x, y), len(valid)

def predict_real_distance(measured_distance: float, direction: str, angle: float) -> float:
    """
    2D ölçümlerde elde edilen mesafenin, 3D'deki gerçek uzunluğa yakın tahminini yapmak için
    kullanılan düzeltme fonksiyonu. Farklı yönlerde (up/down veya left/right) farklı düzeltme
    faktörleri uygulanır.
    
    Öneri:
    - Eğer yön "up" ya da "down" ise, ideal açıyı 90 (veya -90) kabul edip, 
        ölçüm ile gerçek uzunluk arasındaki farkı cos farkı ile telafi ediyoruz.
    - Eğer yön "left" ya da "right" ise, ideal açıyı sıfır (right) veya 180 (left) kabul ediyoruz.
    """
    if angle is None:
        return measured_distance
    
    # Bu örnekte basit bir düzeltme yapılmaktadır:
    # diff, ideal açı ile mevcut angle arasındaki fark (mutlak değer)
    if direction == "up":
        ideal = 90
    elif direction == "down":
        ideal = -90
    elif direction == "right":
        ideal = 0
    elif direction == "left":
        ideal = 180  # 180 veya -180 aynıdır.
    else:
        ideal = 0

    diff = abs(angle - ideal)
    # Çok yüksek farklarda düzeltme faktörünü sınırlıyoruz; 
    # örneğin diff 60 derece ise cos(60)=0.5, böylece ölçüm iki katına çıkabilir.
    # Fakat uç durumları engellemek için en fazla 2 kat düzeltme uygulanmasını sağlayabiliriz.
    factor = 1 / max(math.cos(math.radians(diff)), 0.5)
    return measured_distance * factor / 10

def perpendicular_distance(line_start, line_end, point):
    """Verilen doğruya, point'in dik uzaklığını hesaplar."""
    vx = line_end[0] - line_start[0]
    vy = line_end[1] - line_start[1]
    mag = math.sqrt(vx * vx + vy * vy)
    if mag == 0:
        return None
    vx /= mag
    vy /= mag
    wx = point[0] - line_start[0]
    wy = point[1] - line_start[1]
    proj = wx * vx + wy * vy
    proj_point = (line_start[0] + proj * vx, line_start[1] + proj * vy)
    return compute_distance(point, proj_point)

def point_line_param(line_start, line_end, point):
    """
    line_start ile line_end arasındaki doğru segmentinde, point'in 
    t parametresini hesaplar. (0-1 arasında ise segment üzerinde demektir.)
    """
    vx = line_end[0] - line_start[0]
    vy = line_end[1] - line_start[1]
    wx = point[0] - line_start[0]
    wy = point[1] - line_start[1]
    denom = vx * vx + vy * vy
    if denom == 0:
        return None
    return (wx * vx + wy * vy) / denom

def normalize(vec):
    mag = math.hypot(*vec)
    if mag == 0:
        return (0, 0)
    return (vec[0]/mag, vec[1]/mag)

def average_direction(a, b, c):
    
    # Vektörleri oluştur
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])

    # Ortalama vektör
    avg = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
    return normalize(avg)

def get_vector(a, b):
    """İki nokta arasındaki yön vektörü hesaplanır."""
    start_point = (int(a[0]), int(a[1]))
    end_point = (int(b[0]), int(b[1]))

    return (b[0] - a[0], b[1] - a[1]), start_point, end_point

def are_vectors_approximately_perpendicular(angle1_deg, angle2_deg, tolerance_deg=10):
    # Önce None kontrolü yap
    if angle1_deg is None or angle2_deg is None:
        return False, 0  # Açı eksikse perpendicular diyemeyiz
    
    diff = abs(angle1_deg - angle2_deg) % 180
    normalized_angle = abs(diff - 90)
    return normalized_angle <= tolerance_deg, normalized_angle

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
        x1 += 0.1
    # Eğimi hesapla
    m = (y2 - y1) / (x2 - x1)
    # y-kesişimini hesapla: b = y1 - m * x1
    b = y1 - m * x1
    # Belirtilen x için y değeri
    y = m * x_value + b
    return int(y)

def compute_pooping_score(keypoints, direction, direction_angle, frame):
    """
    Köpeğin pooping pozisyonunu, elimizdeki keypoint verilerinden esnek bir şekilde tahmin eder.
    Koşullar:
    A) Ön ve arka patiler arasındaki mesafe: Bu mesafe, arka bacak (leg) uzunluğunun yarısından
        küçükse pozitif puan.
    B) Tail Start, arka dirsek (Rear Elbow) ile arka diz (Rear Knee) arasına yakınsa,
        pozitif puan.
    C) Kuyruk havadaysa (Tail End'in konumu) negatif katkı.
    D) Ek olarak, infer_dog_direction(keypoints) fonksiyonundan dönen direction ve angle bilgisi
    k   ullanılarak; gerçek mesafe değerleri düzeltilebilsin.
    
    Ayrıca; arka patiler arasındaki mesafe, düzeltildikten sonra eğer hesaplanan
    arka bacak uzunluğunun yarısından fazla ise, skor artırıcı olarak eklenebilir.
    """
    score = 0

    # 0. Köpeğin kafasının orta noktası
    head = []
    if keypoints.get("Nose"): head.append(keypoints.get('Nose'))
    if keypoints.get("Withers"): head.append(keypoints.get('Withers'))
    if keypoints.get("Chin"): head.append(keypoints.get('Chin'))
    if keypoints.get("Left Ear Base"): head.append(keypoints.get('Left Ear Base'))
    if keypoints.get("Right Ear Base"): head.append(keypoints.get('Right Ear Base'))
    if keypoints.get("Left Ear Tip"): head.append(keypoints.get('Left Ear Tip'))
    if keypoints.get("Right Ear Tip"): head.append(keypoints.get('Right Ear Tip'))
    
    avrg_head, n_hp = average_points(head)



    # 1. Ön ve arka patilerin (paws) ortalama noktalarından mesafe hesaplanması
    front_paws = []
    if keypoints.get("Front Left Paw"): front_paws.append(keypoints.get('Front Left Paw'))
    if keypoints.get('Front Right Paw'): front_paws.append(keypoints.get('Front Right Paw'))
    rear_paws =[]
    if keypoints.get("Rear Left Paw"): rear_paws.append(keypoints.get('Rear Left Paw'))
    if keypoints.get('Rear Right Paw'): rear_paws.append(keypoints.get('Rear Right Paw'))


    avrg_point_front_paws, n_front = average_points([
        keypoints.get("Front Left Paw"),
        keypoints.get("Front Right Paw")
    ])
    avrg_point_rear_paws, n_rear = average_points([
        keypoints.get("Rear Left Paw"),
        keypoints.get("Rear Right Paw")
    ])
    
    # Arka bacak uzunluğunu belirleyelim (hesaplanan mesafe = rear elbow -> rear knee)
    front_left_leg_length = None
    if keypoints.get("Front Left Elbow") and keypoints.get("Front Left Knee"):
        front_left_leg_length = compute_distance(keypoints["Front Left Elbow"], keypoints["Front Left Knee"])

    front_right_leg_length = None
    if keypoints.get("Front Right Elbow") and keypoints.get("Front Right Knee"):
        front_right_leg_length = compute_distance(keypoints["Front Right Elbow"], keypoints["Front Right Knee"])

    front_leg_length = None
    count_leg = 0

    if front_left_leg_length is not None:
        front_leg_length = front_left_leg_length
        count_leg += 1

    if front_right_leg_length is not None:
        if front_leg_length is None:
            front_leg_length = front_right_leg_length
        else:
            front_leg_length = (front_leg_length + front_right_leg_length) / 2
        count_leg += 1

    
    # Koşul A: Ön ve arka patiler arasındaki mesafeden puan
    if avrg_point_front_paws is not None and avrg_point_rear_paws is not None and front_left_leg_length is not None:
        measured_front_rear_paws_distance = compute_distance(avrg_point_front_paws, avrg_point_rear_paws)
        # Açı düzeltmesi ile gerçek mesafeyi tahmin et:
        corrected_front_rear_paws_distance = predict_real_distance(measured_front_rear_paws_distance, direction, direction_angle)
        # Arka bacak uzunluğu için de açı düzeltmesi:
        corrected_leg_length = predict_real_distance(front_leg_length, direction, direction_angle)
        
        # Eğer iki pati arasındaki mesafe, düzeltildiğinde arka bacak uzunluğunun yarısından küçükse,
        # pozitif katkı ekleyelim.
        threshold = corrected_leg_length * 2
        if corrected_front_rear_paws_distance < threshold:
            score += 30 ## Ön bacaklarla arka bacaklar arasındaki mesafe darsa +30
            print("Ön patiler ile arka patiler arasindaki mesafe dar +30")
        
        # Ek olarak, eğer arka patiler arasındaki mesafe, arka bacak uzunluğunun yarısından fazlaysa,
        # yani ölçüm beklenenden uzun çıkmışsa, bu da bir ipucu olabilir.
        if len(rear_paws) == 2:
            measured_rear_paws_distance =  compute_distance(rear_paws[0], rear_paws[1])
            corrected_rear_paws_distance = predict_real_distance(measured_rear_paws_distance, direction, direction_angle)
            if corrected_rear_paws_distance > (corrected_leg_length * 0.75):
                weight = corrected_rear_paws_distance - (corrected_leg_length * 0.75)
                score += weight
    

    # Koşul B: Kuyruk başlangıcı (Tail Start) arka dizi (Rear Elbow - Rear Knee) segmenti üzerindeyse     açıya göre ayaklara bak
    tail_start = keypoints.get("Tail Start")
    avrg_elbow, n_elbow = average_points([
        keypoints.get("Rear Left Elbow"),
        keypoints.get("Rear Right Elbow"),
        keypoints.get("Front Left Elbow"),
        keypoints.get("Front Right Elbow")
    ])
    avrg_knee, n_knee = average_points([
        keypoints.get("Rear Left Knee"),
        keypoints.get("Rear Right Knee"),
        keypoints.get("Front Left Knee"),
        keypoints.get("Front Right Knee"),
    ])
    if tail_start is not None and avrg_elbow is not None and avrg_knee is not None:
        t_param = point_line_param(avrg_elbow, avrg_knee, tail_start)
        perp_dist = perpendicular_distance(avrg_elbow, avrg_knee, tail_start)
        leg_line_length = compute_distance(avrg_elbow, avrg_knee)

        
        if t_param is not None and 0 <= t_param <= 1 and perp_dist is not None:
            if perp_dist < leg_line_length * 0.2:
                score += 30
    

        # Koşul C: ön patiler vektörü Köpeğin Vucüduna dik mi? - Arka bacaklar çapraz mı?
    all_legs = []
    front_left_leg = []
    if keypoints.get("Front Left Elbow"): front_left_leg.append(keypoints.get("Front Left Elbow")) 
    if keypoints.get("Front Left Knee"): front_left_leg.append(keypoints.get("Front Left Knee")) 
    if keypoints.get("Front Left Paw"): front_left_leg.append(keypoints.get("Front Left Paw"))
    all_legs.append(front_left_leg)
    
    front_right_leg = []
    if keypoints.get("Front Right Elbow"): front_right_leg.append(keypoints.get("Front Right Elbow")) 
    if keypoints.get("Front Right Knee"): front_right_leg.append(keypoints.get("Front Right Knee")) 
    if keypoints.get("Front Right Paw"): front_right_leg.append(keypoints.get("Front Right Paw"))
    all_legs.append(front_right_leg)
    
    rear_left_leg = []
    if keypoints.get("Rear Left Elbow"): rear_left_leg.append(keypoints.get("Rear Left Elbow")) 
    if keypoints.get("Rear Left Knee"): rear_left_leg.append(keypoints.get("Rear Left Knee")) 
    if keypoints.get("Rear Left Paw"): rear_left_leg.append(keypoints.get("Rear Left Paw"))
    all_legs.append(rear_left_leg)
    
    rear_right_leg = []
    if keypoints.get("Rear Right Elbow"): rear_right_leg.append(keypoints.get("Rear Right Elbow")) 
    if keypoints.get("Rear Right Knee"): rear_right_leg.append(keypoints.get("Rear Right Knee")) 
    if keypoints.get("Rear Right Paw"): rear_right_leg.append(keypoints.get("Rear Right Paw"))
    all_legs.append(rear_right_leg)

    import cv2
    for idx, leg in enumerate(all_legs):
        if len(leg) == 3:
            sp = leg[0]
            ep, n_ep = average_points([leg[1],leg[2]])
            if sp[0] == ep[0]: 
                #Bacaklar tam dik konumda
                score += 5
            else:
                vector_used, sp, ep = get_vector(sp, ep)
                dx, dy = vector_used
                angle = math.degrees(math.atan2(-dy, dx)) 
                vectorsPerpendicular, diklik_farki = are_vectors_approximately_perpendicular(angle, direction_angle, 15)
                if vectorsPerpendicular:
                    color=(0, 145, 140)
                    if idx < 2: ## Ön baklar diklik farkı kadar score ekleme yap 
                        score += diklik_farki
                        print(f"Ön bacaklar dik +{diklik_farki} ")
                    else: ## Arka Bacaklar dikse -30 
                        score -=30 
                        print(f"Arka Bacaklar dik -30 ")
                
                else:
                    color=(0, 0, 255)
                    if idx >= 2:
                        score += diklik_farki ## Arka bacaklar dik değilse
                        print(f"Arka bacaklar dik değil +{diklik_farki}")
                    else:
                        score -= diklik_farki / 2 ## Ön bacaklar dik değilse
                        print(f"Ön bacaklar dik değil -{diklik_farki} / 2")

                """
                Frame için eklenmiştir silinecektir
                """   
                # x = 0 için y
                y_at_0 = compute_y_for_x(sp, ep, 0)
                
# x             = 640 için y
                y_at_640 = compute_y_for_x(sp, ep, 640)
                sp = (0, y_at_0)
                ep = (640, y_at_640)
                
                cv2.arrowedLine(frame, sp, ep, color, thickness=1, tipLength=0.2)
                dx, dy = vector_used
                # atan2 kullanarak açıyı hesaplıyoruz (ekran koordinatında y yukarı negatif)
                angle = math.degrees(math.atan2(-dy, dx))
        cv2.imwrite("results/keypointDetection.jpg", frame)
    

    # Koşul C: Kuyruk havada mı? (Tail End, Tail Start ile kıyaslanıyor)
    tail_end = keypoints.get("Tail End")
    if tail_start is not None and tail_end is not None:
        if tail_end[1] < tail_start[1]:
            score += 10.
            print(f"Kuyruk havada +10")
    return score

def pooping_score(frame, waiting_score, pad_coordinates):
    print('######################## POOPING DETECTION PART ##########################')

    # 1️⃣ Detection işini bir thread olarak başlat
    result_queue = queue.Queue()
    detection_thread = threading.Thread(target=detection_worker, args=(frame, result_queue))
    detection_thread.start()
    
    # 2️⃣ Keypoint'leri tahmin et (YOLO dog pose model)
    keypoints, score, newFrame = keypoint_prediction.keyPointDetection(frame)
    
    # 3️⃣ Pad kontrolünü bir thread olarak başlat
    pad_result_queue = queue.Queue()
    pad_thread = threading.Thread(target=pad_check_thread, args=(keypoints, pad_coordinates, pad_result_queue, 10))
    pad_thread.start()

    # 4️⃣ Ortalama keypoint skoru hesapla
    if score:
        total_score = sum(score.values())
        average_KP_score = total_score / len(score)
        print(f"Avarage Keypoint Score: {average_KP_score:.4f}")
    else:
        average_KP_score = 0
        print("⚠️  Uyari: Score verisi boş geldi. Ortalama 0 olarak atandı. ⚠️ ")

    # 5️⃣ Yön tahmini
    direction, angle, newFrame = directionPrediction.infer_dog_direction(keypoints, newFrame)
    if direction in ["up", "down"]:
        adjusted_angle = 0
        print(f"⚠️  Uyari: Hayvanin yönü {direction} olduğundan angle 0 olarak değiştirilmiştir. ⚠️ ")
    else:
        adjusted_angle = angle
    print(f"Köpeğin yönü: {direction}, Açi: {adjusted_angle}")

    # 6️⃣ Pooping skoru hesapla
    keypoint_position_score = compute_pooping_score(keypoints, direction, adjusted_angle, newFrame)
    print(Fore.LIGHTMAGENTA_EX + f"Köpeğin Pooping Skoru: {keypoint_position_score}" + Fore.WHITE)

    # 7️⃣ Thread'lerin bitmesini bekle
    detection_thread.join()
    pad_thread.join()

    # 8️⃣ Sonuçları al
    opScore, pooping = result_queue.get()
    print(Fore.LIGHTGREEN_EX + f"Object Detection Pooping Score:{opScore} " + Fore.WHITE)

    pad_result = pad_result_queue.get()
    print(Fore.LIGHTCYAN_EX + f"Köpek pad üstünde mi? {pad_result} Bekleme Süresi: {waiting_score}" + Fore.WHITE)

    # 9️⃣ Son karar
    if direction == "unknown":
        keypoint_position_score += 1
        print(Fore.YELLOW + "⚠️  Direction unknown olduğundan keypoint_position_score +1 ⚠️" + Fore.WHITE)

    if opScore is None:
        opScore = 0  # veya senin belirlediğin default bir değer
    
    final_score = keypoint_position_score * average_KP_score + opScore
    if pooping and final_score > config.pooping_treshold:
        if pad_result:
            set_angle(20)
        else:
            print("ses çaldı")
        
        return True
    else:
        print(Fore.RED + "Köpek büyük ihtimalle pooping pozisyonunda değil.")
        print(Fore.WHITE + '#########################################################################')
        return False
