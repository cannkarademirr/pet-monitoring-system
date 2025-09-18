from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict
import requests
from pathlib import Path
import os
import subprocess

app = FastAPI()

# Cihazlar: device_id -> base_url
device_streams: Dict[str, str] = {}

users = {
    "ck": {"password": "123", "device_id": "smartBowl_001"},
    "raspi": {"password": "123", "device_id": "smartBowl_002"},
}

class DeviceRegister(BaseModel):
    device_id: str
    stream_url: str

class UserLogin(BaseModel):
    username: str
    password: str

class DeviceCommand(BaseModel):
    device_id: str

class PoopingDetectionUpdate(BaseModel):
    device_id: str
    enabled: bool

@app.get("/get_device_info/{device_id}")
async def get_device_info(device_id: str):
    stream_url = device_streams.get(device_id)
    if not stream_url:
        raise HTTPException(status_code=404, detail="Stream URL not found")
    base_url = stream_url.replace("/mjpeg", "")
    return {
        "stream_url": stream_url,
        "raspi_base_url": base_url
    }

@app.post("/register_device")
async def register_device(data: DeviceRegister):
    base_url = data.stream_url.rstrip('/').replace('/mjpeg', '')
    device_streams[data.device_id] = base_url
    print(f"[INFO] Device registered: {data.device_id} -> {base_url}")
    return {"message": "Device registered", "device_id": data.device_id}

@app.get("/get_stream_url/{device_id}")
async def get_stream_url(device_id: str):
    base_url = device_streams.get(device_id)
    if base_url:
        return {"stream_url": f"{base_url}/mjpeg"}
    else:
        raise HTTPException(status_code=404, detail="Device not found")

@app.post("/login")
async def login(user: UserLogin):
    user_info = users.get(user.username)
    if not user_info or user_info["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"device_id": user_info["device_id"]}

@app.post("/take_snapshot")
async def take_snapshot(data: DeviceCommand):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(f"{base_url}/snapshot")
        return {"message": "Snapshot command sent", "status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending snapshot command: {e}")

@app.post("/start_recording")
async def start_recording(data: DeviceCommand):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(f"{base_url}/start_recording")
        return {"message": "Start recording command sent", "status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending start recording command: {e}")

@app.post("/play_audio")
async def play_audio(data: DeviceCommand):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(f"{base_url}/play_audio")
        return {"message": "Play audio command sent", "status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending play audio command: {e}")

@app.get("/get_pooping_detection")
async def get_pooping_detection(device_id: str = Query(...)):
    base_url = device_streams.get(device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.get(f"{base_url}/pooping_detection_status")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting device: {e}")

@app.post("/set_pooping_detection")
async def set_pooping_detection(data: PoopingDetectionUpdate):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(
            f"{base_url}/pooping_detection_toggle",
            json={"enabled": data.enabled}
        )
        return {
            "message": "Command forwarded to device",
            "status_code": response.status_code,
            "raspi_response": response.json()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding command: {e}")

@app.get("/get_snapshots")
async def get_snapshots(device_id: str = Query(...)):
    base_url = device_streams.get(device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.get(f"{base_url}/list_snapshots")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to get snapshots from device")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting device: {e}")

@app.get("/get_recordings")
async def get_recordings(device_id: str = Query(...)):
    base_url = device_streams.get(device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.get(f"{base_url}/list_recordings")
        if response.status_code == 200:
            return response.json()
        else:
            raise HTTPException(status_code=500, detail="Failed to get recordings from device")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting device: {e}")

@app.get("/recordings/{filename}")
async def get_video_file(filename: str):
    file_path = f"./records/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)

@app.get("/video_thumbnail/{device_id}/{filename}")
async def get_video_thumbnail(device_id: str, filename: str):
    base_name = os.path.splitext(filename)[0]
    thumb_path = f"./thumbnails/{base_name}.jpg"
    video_path = f"./records/{filename}"

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    if not os.path.exists(thumb_path):
        os.makedirs("thumbnails", exist_ok=True)
        try:
            subprocess.run([
                "ffmpeg", "-i", video_path,
                "-ss", "00:00:01.000",
                "-vframes", "1",
                thumb_path
            ], check=True)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Thumbnail creation failed: {e}")

    return FileResponse(thumb_path, media_type="image/jpeg")

@app.post("/give_treat")
async def give_treat(data: DeviceCommand):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(f"{base_url}/give_treat")
        return {"message": "Treat command sent", "status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending treat command: {e}")

@app.post("/give_food")
async def give_food(data: DeviceCommand):
    base_url = device_streams.get(data.device_id)
    if not base_url:
        raise HTTPException(status_code=404, detail="Device not found")
    try:
        response = requests.post(f"{base_url}/give_food")
        return {"message": "Food command sent", "status_code": response.status_code}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending food command: {e}")

