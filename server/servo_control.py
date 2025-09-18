# servo_control.py
from gpiozero import AngularServo
from time import sleep

try:
    servo = AngularServo(
        14,
        min_angle=0,
        max_angle=360,
        min_pulse_width=0.0006,
        max_pulse_width=0.0024,
        initial_angle=None
    )
except Exception as e:
    print(f"[ERROR] Servo init failed: {e}")
    servo = None

def set_angle(angle):
    try:
        if servo is None:
            print("[ERROR] Servo object is None")
            return

        print(f"[INFO] Moving servo to {angle}")
        servo.angle = angle * 2
        sleep(3)
        servo.angle = 180
        sleep(3)
        print("[INFO] Servo reset to 180")
        
        servo.angle = None
    except Exception as e:
        print(f"[ERROR] servo_control failed: {e}")
        raise e
