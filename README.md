# Pet Monitoring System

## Overview  
The **Pet Monitoring System** is an AI-powered smart feeding solution designed for pet owners. Built on **Raspberry Pi 5**, it integrates computer vision, IoT, and mobile technologies to remotely monitor pets, automate toilet training, and provide reward-based feeding.  

The system combines:  
- A **smart feeding bowl** with real-time AI inference  
- A **Python FastAPI server** for secure communication  
- A **Flutter mobile app** for remote monitoring and control  

---

## Features  
- 📷 **Live Video Streaming** via Raspberry Pi camera  
- 🤖 **AI-powered Behavior Recognition** using YOLOv11 Nano models:  
  - Object detection (cats & dogs)  
  - Defecation detection  
  - Keypoint-based posture analysis  
- 🍖 **Automatic Reward Feeding** when correct behavior is detected  
- 📱 **Mobile App (Flutter)** for:  
  - Live view  
  - Manual feeding (normal & reward)  
  - Snapshots and video recording  
  - Defining training pad area  
- 🔒 **Secure Communication** via FastAPI + WebSocket  
- ⚡ **Low-latency performance** (<1.5s reward delivery)  

---

## System Architecture  
1. **Raspberry Pi 5** – runs AI models, controls servo motor, streams video.  
2. **Camera Module** – provides real-time video feed.  
3. **Servo Motor** – dispenses food as a reward.  
4. **FastAPI Server** – manages communication, authentication, and command routing.  
5. **Flutter Mobile App** – enables user interaction and remote control.  

---

## Hardware Requirements  
- Raspberry Pi 5 (4GB RAM or higher)  
- Raspberry Pi Camera Module 3  
- SG90 Servo Motor + 3D-printed bowl mechanism  
- 16GB+ microSD card  
- Stable internet connection  

---

## Software Requirements  
- **Raspberry Pi OS Lite (64-bit)**  
- **Python 3.11+**  
- **FastAPI** (backend server)  
- **PyTorch + Ultralytics YOLOv11 Nano** (AI inference)  
- **OpenCV, NumPy, RPi.GPIO** (processing & hardware control)  
- **Flutter SDK** (mobile application)  

---

## Performance  
- **Detection Accuracy (F1 Score):** 0.987  
- **Response Time:** ~1.1 seconds from detection to reward  
- **Stable Operation:** 12+ hours continuous runtime  

---

## Future Work  
- Multi-pet support (cat-specific models)  
- Real-time mobile notifications  
- Night-vision capability  
- Audio-based reinforcement feedback  
- Visual overlays for improved transparency in decision-making  

---

## Project Documents  
- [Poster](docs/poster.png)  
- [Report](docs/report.pdf)  

---

⚡ *This project was developed as a graduation project at Yeditepe University, Department of Computer Engineering.*  
