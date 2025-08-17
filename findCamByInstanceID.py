import subprocess
import os

def run_powershell_command():
    # PowerShell 命令，保存输出到 usb_devices.txt 文件
    cmd = [
        "powershell",
        "-Command",
        "Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB*' } | Select-Object InstanceId, FriendlyName, Class, Status | Out-File 'usb_devices.txt' -Encoding utf8"
    ]
    subprocess.run(cmd, capture_output=True, text=True, shell=True)

def get_usb_device_list_from_file(file_path):
    devices = []
    
    # 打开并读取文件
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    
    # 解析文件中的每一行，提取设备信息
    for line in lines:
        # 过滤包含 'Camera' 的设备
        if "Camera" in line:
            parts = line.split()
            if len(parts) > 3:
                # 提取 InstanceId 和设备名称
                instance_id = parts[0]
                device_name = " ".join(parts[1:-2])  # 去掉状态和 InstanceId
                devices.append({"name": device_name, "instance_id": instance_id})
    
    return devices

def find_camera_by_instance_id(instance_id, devices):
    # 查找匹配的设备
    for camera in devices:
        if camera["instance_id"] == instance_id:
            print(f"找到摄像头: {camera['name']}，InstanceId: {camera['instance_id']}")
            return camera
    print(f"没有找到匹配的摄像头: {instance_id}")
    return None

def main():
    # 运行 PowerShell 命令，将设备信息保存到 usb_devices.txt
    run_powershell_command()
    
    # 文件路径
    file_path = "usb_devices.txt"  # 确保文件路径正确
    
    # 读取并解析设备列表
    devices = get_usb_device_list_from_file(file_path)
    
    # 输入需要选择的摄像头 InstanceId
    # target_instance_id = "USB\\VID_05A3&PID_9230&MI_00\\6&17A10D50&0&0000"  # 替换成目标摄像头的 InstanceId

    target_instance_id = "USB\VID_1E4E&PID_0109&MI_00\\6&6C28DBB&1&0000"

     
    camera = find_camera_by_instance_id(target_instance_id, devices)
    
    if camera:
        print(f"找到并选择摄像头: {camera['name']}，InstanceId: {camera['instance_id']}")
    else:
        print("未能找到指定的摄像头.")

if __name__ == "__main__":
    main()
    
