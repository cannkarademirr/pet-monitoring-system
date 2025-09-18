from collections import defaultdict, deque
import multiprocessing
import os
import random
import threading
import cv2
from flask import Flask, Response, jsonify, request, send_from_directory
from matplotlib import colors
import numpy as np
from ultralytics import YOLO
from picamera2 import Picamera2
import time
import json

# Kendi Kütüphanelerim
import config
import stationary_detector
from poopingDetection import pooping_score
from servo_control import set_angle

# Kamerayı başlat
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    raw={"size": (1640, 1232)},
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()
time.sleep(2)  # Kamera açıldıktan sonra buffer dolması için bekle
import requests
def register_device(server_url):
    # Tünel URL üzerinden MJPEG yayın
    stream_url = f"{TUNNEL_URL}/mjpeg"

    data = {
        "device_id": DEVICE_ID,
        "stream_url": stream_url
    }

    try:
        response = requests.post(f"{server_url}/register_device", json=data)
        if response.status_code == 200:
            print(f"[INFO] Device {DEVICE_ID} registered successfully!")
        else:
            print(f"[ERROR] Registration failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Registration exception: {e}")

# Cihaz ID (Her cihaz için farklı)
DEVICE_ID = "smartBowl_002"

# Cihazın localini tünelleyen URL
TUNNEL_URL = "https://e8bc-31-223-13-183.ngrok-free.app"

# Server (merkezi) URLS
SERVER_URL = "https://61ff-176-234-128-30.ngrok-free.app"
register_device(SERVER_URL)
# Önceki 30 frame için buffer
prev_frames = deque(maxlen = int(config.recorded_video_frames * config.recorded_video_seconds / 2))
recording = False
frames_after = 0
out = None
frame = None
# Global variables for fps calculation and video saving
total_fps = 0
average_fps = 0
num_of_frame = 0
video_frames = []
model = YOLO("models/yolo11n.pt")  # ADRESİ DÜZENLE !
labels = model.names
colors = [[random.randint(0, 255) for _ in range(0, 3)] for _ in labels]
pooping_detection = False
object_history = defaultdict(list)  # Nesne id’sine göre geçmiş merkez pozisyonları

# --- Parametreler -----------------------------------------------------------
INITIAL_THRESHOLD   = 1          # ilk tetik eşiği
GROWTH_FACTOR       = 10          # her tetik sonrası eşiği kaç kat büyüteceğiz
# ---------------------------------------------------------------------------

# Her obje (class_id) için durum tabloları
trigger_counter   = defaultdict(int)                # kaç kere 'sabit' gördük
trigger_threshold = defaultdict(lambda: INITIAL_THRESHOLD)
object_center = None
app = Flask(__name__)
def draw_transparent_pad(frame, pad_polygon, color=(0, 255, 0), alpha=0.3):
    """
    frame: BGR görüntü (numpy array)
    pad_polygon: [(x1, y1), (x2, y2), ...]
    color: BGR rengi (örn: (0,255,0) yeşil)
    alpha: dolgu opaklığı (0=şeffaf, 1=opak)        SİLİNECEK PADİ GÖRMEK İÇİN ÇİZİLDİ
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(pad_polygon, dtype=np.int32)], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def load_pad_coordinates_from_json(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            pad_coordinates = data.get("pad_coordinates")
            if pad_coordinates and isinstance(pad_coordinates, list):
                print(f"Pad koordinatları yüklendi: {pad_coordinates}")
                return [tuple(coord) for coord in pad_coordinates]
            else:
                print("JSON dosyasında pad_coordinates bulunamadı veya format hatalı. pad_coordinates = None")
                return None
    except FileNotFoundError:
        print(f"Hata: {file_path} bulunamadı. pad_coordinates = None")
        return None
    except json.JSONDecodeError:
        print("Hata: JSON formatı hatalı. pad_coordinates = None")
        return None
    except Exception as e:
        print(f"Hata oluştu: {e}. pad_coordinates = None")
        return None

pad_coordinates = load_pad_coordinates_from_json("config.json")

def should_fire(class_id):
    """Bu obje sabit kalmaya devam ederken, tetikleme zamani geldi mi?"""
    trigger_counter[class_id] += 1

    if trigger_counter[class_id] >= trigger_threshold[class_id]:
        trigger_counter[class_id] = 0
        trigger_threshold[class_id] *= GROWTH_FACTOR
        print(f"Object ID: {class_id}, Threshold: {trigger_threshold[class_id]}")
        return True
    return False

def reset_object(class_id):
    """Obje hareket edince sayaç eşiği başa sar."""
    trigger_counter[class_id]   = 0
    trigger_threshold[class_id] = INITIAL_THRESHOLD

def process_frame(frame, class_id):
    global recording, trigger_threshold

    if  not recording and pooping_score(frame, trigger_threshold[class_id], pad_coordinates):
        recording = True
        save_video(frame, f"records/pooping_{int(time.time())}.mp4")
def save_video(frame, filename):
    global recording, prev_frames
    if not list(prev_frames):
        return
    height, width, _ = prev_frames[0].shape
    filename = filename.replace('.avi', '.mp4')  # 🔥 mp4 uzantısı kullan
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), config.recorded_video_frames, (width, height))
def generate_frames():
    while True:
        global stream_frame
        ret, jpeg = cv2.imencode('.jpg', stream_frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/mjpeg')
def mjpeg_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

import requests
def register_device(server_url):
    # Tünel URL üzerinden MJPEG yayın
    stream_url = f"{TUNNEL_URL}/mjpeg"

    data = {
        "device_id": DEVICE_ID,
        "stream_url": stream_url
    }

    try:
        response = requests.post(f"{server_url}/register_device", json=data)
        if response.status_code == 200:
            print(f"[INFO] Device {DEVICE_ID} registered successfully!")
        else:
            print(f"[ERROR] Registration failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Registration exception: {e}")


@app.route('/snapshot', methods=['POST'])
def snapshot():
    global stream_frame  # 🌟 Global frame değişkenini kullanıyoruz
    if stream_frame is None:
        return jsonify({"status": "error", "message": "Mevcut frame yok"}), 500
    filename = f'snapshot_{int(time.time())}.jpg'
    filepath = os.path.join('snapshots', filename)
    os.makedirs('snapshots', exist_ok=True)
    cv2.imwrite(filepath, stream_frame)
    print(f"Snapshot kaydedildi: {filepath}")
    return jsonify({"status": "success", "filename": filename})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording, video_writer, manual_recording, stream_frame

    if not recording:
        if stream_frame is None:
            return jsonify({"status": "error", "message": "Mevcut stream_frame yok"}), 500

        os.makedirs('recordings', exist_ok=True)
        filename = f'recordings/video_{int(time.time())}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        height, width = stream_frame.shape[:2]
        video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
        recording = True
        manual_recording = True

        def record():
            global recording, manual_recording, video_writer, stream_frame
            while manual_recording:
                if stream_frame is not None:
                    video_writer.write(stream_frame)
                time.sleep(1 / fps)
            video_writer.release()
            recording = False


        threading.Thread(target=record).start()
        return jsonify({"status": "success", "message": "Kayıt başlatıldı"})

    else:
        # Kayıt durdurma
        manual_recording = False
        return jsonify({"status": "success", "message": "Kayıt durduruldu"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    if recording:
        recording = False
        print("Kayıt durduruldu")
        return jsonify({"status": "success", "message": "Kayıt durduruldu"})
    else:
        return jsonify({"status": "error", "message": "Kayıt zaten durmuş"})

@app.route('/play_audio', methods=['POST'])
def play_audio():
    # Basit bir bip sesi çalma (veya ses dosyası)
    try:
        os.system('aplay /usr/share/sounds/alsa/Front_Center.wav')
        print("Ses çalındı")
        return jsonify({"status": "success", "message": "Ses çalındı"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

SNAPSHOT_DIR = 'snapshots'
RECORDING_DIR = 'recordings'
THUMBNAIL_DIR = 'thumbnails'

@app.route('/get_snapshots')
def get_snapshots_route():
    device_id = request.args.get('device_id')  # kullanmıyoruz ama frontend için var
    return list_snapshots()

def list_snapshots():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    return jsonify(files)

@app.route('/get_recordings')
def get_recordings_route():
    device_id = request.args.get('device_id')
    return list_recordings()

def list_recordings():
    os.makedirs(RECORDING_DIR, exist_ok=True)
    files = [f for f in os.listdir(RECORDING_DIR) if f.endswith('.mp4')]
    return jsonify(files)

@app.route('/snapshots/<path:filename>')
def serve_snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)

@app.route('/recordings/<path:filename>')
def serve_recording(filename):
    return send_from_directory(RECORDING_DIR, filename)
def start_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
@app.route('/pooping_detection_status', methods=['GET'])
def get_pooping_detection_status():
    return jsonify({"enabled": pooping_detection})


@app.route('/pooping_detection_toggle', methods=['POST'])
def set_pooping_detection_status():
    global pooping_detection
    try:
        data = request.get_json()
        pooping_detection = data.get("enabled", False)
        print(f"[INFO] pooping_detection set to {pooping_detection}")
        return jsonify({"status": "success", "enabled": pooping_detection})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/give_treat', methods=['POST'])
def give_treat():
    try:
        print("🎉 Reward treat dispensed!")
        set_angle(20)
        return jsonify({"status": "success", "message": "Treat dispensed"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/give_food', methods=['POST'])
def give_food():
    try:
        print("?? Normal food dispensed!")
        set_angle(160)
        return jsonify({"status": "success", "message": "Food dispensed"})
    except Exception as e:
        import traceback
        print("[ERROR] /give_food exception occurred:")
        traceback.print_exc()  # FULL traceback yazd?r?r
        return jsonify({"status": "error", "message": str(e)}), 500

multiprocessing.freeze_support()
flask_thread = threading.Thread(target=start_flask)
flask_thread.daemon = True
flask_thread.start()
pooping_triger = 0
while True:
    start = time.time()
    key = cv2.waitKey(20) & 0xFF
    frame = picam2.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    prev_frames.append(frame.copy()) ## Video Kayıt
    stream_frame = frame.copy()
# YOLO tahmini sadece ekran ya da video/kamera fark etmeksizin yapılır
    if pooping_detection and pooping_triger == 3:
        pooping_triger = 0
        results = model(frame, verbose=False)[0]
        boxes = np.array(results.boxes.data.tolist())
        for box in boxes:
            x1, y1, x2, y2, score, class_id = box
            x1, y1, x2, y2, class_id = int(x1), int(y1), int(x2), int(y2), int(class_id)
            box_color = colors[class_id]
            if score > config.confidence_score:
                object_center = stationary_detector.get_center(x1, y1, x2, y2)

                object_history[class_id].append(object_center)
                if class_id == 15 or class_id == 16:
                    if stationary_detector.is_stationary(object_history[class_id]):     # Obje sabit
                        #print("Obje belli bir süredir sabit")
                        if should_fire(class_id):
                            stationary_detector.on_stationary_object_detected(class_id, class_id, object_center)
                            threading.Thread(target=process_frame, args=(frame, class_id)).start() ##frame cap ve capture method video kaydı için eklenmiştir.
                    else:       # Obje hareket etti → sayaçları sıfırla
                        reset_object(class_id)
                        print ("Obje Haraketli")
                # Çizim işlemleri burada kalabilir
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[class_id], 2)
                class_name = results.names[class_id]
                text = f"{class_name}: %{score * 100:.2f}"
                text_loc = (x1, y1-10)
                labelSize, baseLine = cv2.getTextSize(text, config.font, 1, 1)
                cv2.rectangle(frame, 
                            (x1, y1 - 10 - labelSize[1]), 
                            (x1 + labelSize[0], int(y1 + baseLine-10)), 
                            colors[class_id], 
                            cv2.FILLED)
                cv2.putText(frame, text, text_loc, config.font, 1, config.text_color_w, thickness=1)
    pooping_triger += 1
    end = time.time()
    num_of_frame += 1
    fps = 1 / (end-start)
    total_fps = total_fps + fps
    average_fps = total_fps / num_of_frame
    avg_fps = float("{:.2f}".format(average_fps))
    if pad_coordinates:
        draw_transparent_pad(frame, pad_coordinates, color=(0, 255, 0), alpha=0.3)
    
    cv2.rectangle(frame, (10,2), (280,50), (0,255,0), -1)
    cv2.putText(frame, "FPS: "+ str(avg_fps), (20,40), config.font, 1.5, config.text_color_b, thickness=3)
    #cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamera kapatılır
picam2.stop()
picam2.close()
cv2.destroyAllWindows()
