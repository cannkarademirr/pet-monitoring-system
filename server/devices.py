from asyncio import Lock

connected_devices = {}
device_locks = {}
device_ips = {}  # 👈 cihazların IP adreslerini sakla

def register_device(device_id: str, websocket, ip_address: str):
    connected_devices[device_id] = websocket
    device_ips[device_id] = ip_address
    if device_id not in device_locks:
        device_locks[device_id] = Lock()

def unregister_device(device_id: str):
    connected_devices.pop(device_id, None)
    device_locks.pop(device_id, None)
    device_ips.pop(device_id, None)

def get_device(device_id: str):
    return connected_devices.get(device_id)

def get_device_lock(device_id: str):
    return device_locks.get(device_id)

def get_device_ip(device_id: str):
    return device_ips.get(device_id)
