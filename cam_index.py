import comtypes
import comtypes.client
import numpy as np
from ctypes import *
from comtypes import GUID
import cv2

# 安装: pip install comtypes

class DirectShowCamera:
    def __init__(self):
        comtypes.CoInitialize()
        self.graph_builder = None
        self.capture_builder = None
        self.media_control = None
        self.sample_grabber = None
        
    def list_cameras_with_instance_id(self):
        """列出所有摄像头及其InstanceID"""
        import win32com.client
        wmi = win32com.client.GetObject("winmgmts:")
        cameras = []
        
        for device in wmi.InstancesOf("Win32_PnPEntity"):
            if (device.Name and 
                any(keyword in device.Name.lower() for keyword in ['camera', 'webcam']) and
                device.Present):
                cameras.append({
                    'name': device.Name,
                    'instance_id': device.PNPDeviceID,
                    'device_id': device.DeviceID
                })
        return cameras
    
    def open_by_instance_id(self, instance_id):
        """通过InstanceID打开摄像头"""
        try:
            # 创建FilterGraph
            self.graph_builder = comtypes.client.CreateObject("FilterGraph.FilterGraph")
            
            # 创建Capture Graph Builder
            self.capture_builder = comtypes.client.CreateObject("CaptureGraphBuilder2.CaptureGraphBuilder2")
            self.capture_builder.SetFiltergraph(self.graph_builder)
            
            # 枚举视频输入设备
            system_device_enum = comtypes.client.CreateObject("SystemDeviceEnum.SystemDeviceEnum")
            video_input_enum = system_device_enum.CreateClassEnumerator("{860BB310-5D01-11D0-BD3B-00A0C911CE86}", 0)
            
            # 查找匹配的设备
            source_filter = None
            moniker = comtypes.POINTER(comtypes.IUnknown)()
            
            while video_input_enum.Next(1, byref(moniker), None) == 0:
                property_bag = moniker.BindToStorage(None, None, comtypes.IID_IPropertyBag)
                
                # 获取设备路径
                variant = comtypes.VARIANT()
                property_bag.Read("DevicePath", byref(variant), None)
                device_path = variant.value
                
                # 检查是否匹配InstanceID
                if instance_id.lower() in device_path.lower():
                    # 创建设备过滤器
                    source_filter = moniker.BindToObject(None, None, comtypes.IID_IBaseFilter)
                    break
            
            if not source_filter:
                return False
                
            # 添加源过滤器到图形
            self.graph_builder.AddFilter(source_filter, "Video Source")
            
            # 创建Sample Grabber
            self.sample_grabber = comtypes.client.CreateObject("SampleGrabber.SampleGrabber")
            sample_grabber_filter = self.sample_grabber.QueryInterface(comtypes.IID_IBaseFilter)
            self.graph_builder.AddFilter(sample_grabber_filter, "Sample Grabber")
            
            # 设置媒体类型
            media_type = comtypes.GUID("{73646976-0000-0010-8000-00AA00389B71}")  # RGB24
            self.sample_grabber.SetMediaType(media_type)
            
            # 连接过滤器
            self.capture_builder.RenderStream(None, None, source_filter, sample_grabber_filter, None)
            
            # 获取媒体控制接口
            self.media_control = self.graph_builder.QueryInterface(comtypes.IID_IMediaControl)
            
            # 开始捕获
            self.media_control.Run()
            
            return True
            
        except Exception as e:
            print(f"打开摄像头失败: {e}")
            return False
    
    def get_frame(self):
        """获取一帧图像"""
        if not self.sample_grabber:
            return None
            
        try:
            # 获取当前样本
            buffer_size = self.sample_grabber.GetCurrentBuffer(0, None)
            if buffer_size <= 0:
                return None
                
            buffer = (c_byte * buffer_size)()
            self.sample_grabber.GetCurrentBuffer(buffer_size, buffer)
            
            # 转换为numpy数组 (这里需要根据实际格式调整)
            frame_data = np.frombuffer(buffer, dtype=np.uint8)
            
            # 假设是640x480 RGB格式，实际使用时需要获取真实尺寸
            if len(frame_data) >= 640 * 480 * 3:
                frame = frame_data[:640*480*3].reshape((480, 640, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
                
        except Exception as e:
            print(f"获取帧失败: {e}")
            
        return None
    
    def release(self):
        """释放资源"""
        if self.media_control:
            self.media_control.Stop()
        comtypes.CoUninitialize()

# 使用示例
camera = DirectShowCamera()
cameras = camera.list_cameras_with_instance_id()

print("可用摄像头:")
for i, cam in enumerate(cameras):
    print(f"{i}: {cam['name']}")
    print(f"   InstanceID: {cam['instance_id']}")

# 选择第一个摄像头
if cameras:
    target_instance_id = cameras[0]['instance_id']
    if camera.open_by_instance_id(target_instance_id):
        print("成功打开摄像头!")
        
        import time
        for _ in range(100):
            frame = camera.get_frame()
            if frame is not None:
                cv2.imshow('Camera', frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
            time.sleep(0.033)
        
        camera.release()
        cv2.destroyAllWindows()