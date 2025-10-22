import serial
import time
import threading

# 设备常量（根据现场需要可调整）
MODBUS_SLAVE_ID = 0x09  # Robotiq 默认常见为 0x09
START_REG = 0x03E8      # 起始寄存器地址 1000
WRITE_MULTI = 0x10      # 功能码：写多个保持寄存器
READ_HOLDING = 0x03     # 功能码：读保持寄存器（Robotiq 状态通常在 0x07D0 起始的输入/保持寄存器）
STATUS_START_REG = 0x07D0  # 2000，常见为状态起始寄存器

class Robotiq2F85:
    def __init__(self, port='COM5', baudrate=115200, timeout=2.0, debug: bool = True):
        """初始化夹爪连接
        timeout: 串口读超时（秒）
        debug:   打印发送/接收帧，便于排查
        """
        self.debug = debug
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE)
        # 清空缓冲，给设备一点时间
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass
        time.sleep(0.2)
        print(f"已连接到 {port}，baud={baudrate}, timeout={timeout}s")
    
    # =============== 内部工具函数 ===============
    @staticmethod
    def _crc16_modbus(data: bytes) -> int:
        """计算 Modbus CRC16（多项式 0xA001，返回 16 位整数）。"""
        crc = 0xFFFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc & 0xFFFF

    def _build_frame(self, data6: list[int]) -> bytes:
        """构造完整 Modbus RTU 帧（写 3 个寄存器=6 字节数据）。
        data6: 长度为 6 的数据区字节列表。
        帧格式： [id, 0x10, regHi, regLo, 0x00, 0x03, 0x06, data6..., CRCLo, CRCHi]
        """
        if len(data6) != 6:
            raise ValueError("data6 必须是 6 个字节")

        frame_wo_crc = bytes([
            MODBUS_SLAVE_ID,
            WRITE_MULTI,
            (START_REG >> 8) & 0xFF,
            START_REG & 0xFF,
            0x00, 0x03,
            0x06,
            *data6,
        ])
        crc = self._crc16_modbus(frame_wo_crc)
        crc_lo = crc & 0xFF
        crc_hi = (crc >> 8) & 0xFF
        return frame_wo_crc + bytes([crc_lo, crc_hi])

    def _build_read_frame(self, start_reg: int, quantity: int) -> bytes:
        """构造 Modbus 读寄存器帧（功能码 0x03）。"""
        frame_wo_crc = bytes([
            MODBUS_SLAVE_ID,
            READ_HOLDING,
            (start_reg >> 8) & 0xFF,
            start_reg & 0xFF,
            (quantity >> 8) & 0xFF,
            quantity & 0xFF,
        ])
        crc = self._crc16_modbus(frame_wo_crc)
        return frame_wo_crc + bytes([crc & 0xFF, (crc >> 8) & 0xFF])

    def _send_command(self, data6: list[int], wait_s: float = 0.2) -> bytes:
        """发送命令并尝试读取标准响应（写多个寄存器通常回 8 字节）。"""
        frame = self._build_frame(data6)
        if self.debug:
            print("TX:", ' '.join(f"{b:02X}" for b in frame))
        # 先清输入，避免读到旧数据
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        # 发送
        self.ser.write(frame)
        try:
            self.ser.flush()
        except Exception:
            pass
        # 写入后稍等，从站回包一般是 8 字节：[id, 0x10, regHi, regLo, 0x00, 0x03, CRCLo, CRCHi]
        time.sleep(wait_s)
        # 读固定长度或尽量多读一点
        resp = self.ser.read(8)
        if self.debug:
            if resp:
                print("RX:", ' '.join(f"{b:02X}" for b in resp), f"(len={len(resp)})")
            else:
                print("RX: <no data>")
        return resp

    @staticmethod
    def _clamp_byte(x: int) -> int:
        return max(0, min(255, int(x)))

    # =============== 业务操作 ===============
    def activate(self):
        """激活夹爪（先复位清除故障，再激活，轮询确认 gACT=1）"""
        print("激活夹爪...")
        # 步骤1：复位（rACT=0），清除故障位
        print("  [1/3] 复位清除故障...")
        self._send_command([0x00, 0x00, 0x00, 0x00, 0x00, 0x00], wait_s=0.2)
        time.sleep(0.5)
        
        # 步骤2：激活（rACT=1）
        print("  [2/3] 发送激活指令...")
        self._send_command([0x01, 0x00, 0x00, 0x00, 0x00, 0x00], wait_s=0.2)
        time.sleep(0.5)
        
        # 步骤3：轮询状态，等待 gACT=1（激活完成）
        print("  [3/3] 等待激活完成（gACT=1）...")
        for attempt in range(10):
            time.sleep(0.3)
            resp = self.read_status()
            if len(resp) >= 4:
                gACT = (resp[3] >> 0) & 0x01  # Byte0 的 bit0
                gSTA = (resp[3] >> 4) & 0x03  # Byte0 的 bit4-5
                gFLT = (resp[3] >> 0) & 0x0F  # Byte0 的 bit0-3（完整故障码）
                if self.debug:
                    print(f"    尝试 {attempt+1}: gACT={gACT}, gSTA={gSTA}, gFLT={gFLT:02X}")
                if gACT == 1:
                    print("激活成功！")
                    return
        print("警告：激活超时，gACT 仍为 0。请检查供电/接线或手动复位夹爪。")

    def read_status(self, regs: int = 3, verbose: bool = None):
        """读取状态寄存器，默认读取 3 个寄存器（6 字节），仅做原始打印。
        verbose: 是否打印详细 TX/RX，默认跟随 self.debug
        """
        if verbose is None:
            verbose = self.debug
        frame = self._build_read_frame(STATUS_START_REG, regs)
        if verbose:
            print("TX (RD):", ' '.join(f"{b:02X}" for b in frame))
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.ser.write(frame)
        try:
            self.ser.flush()
        except Exception:
            pass
        # 期望: [id, 0x03, byteCount, data..., CRClo, CRChi]
        resp = self.ser.read(64)
        if verbose:
            if resp:
                print("RX (RD):", ' '.join(f"{b:02X}" for b in resp), f"(len={len(resp)})")
            else:
                print("RX (RD): <no data>")
        return resp
    
    def get_position(self) -> int:
        """读取当前夹爪位置 (0-255, 0=完全打开, 255=完全关闭)。
        返回 -1 表示读取失败。
        """
        resp = self.read_status(regs=3, verbose=False)
        if len(resp) >= 7:
            # 状态帧格式：[id, 0x03, byteCount, Byte0, Byte1, Byte2, Byte3, ...]
            # Byte3 是当前位置 gPO (position object)
            gPO = resp[6]  # 索引 6 = Byte3
            return gPO
        return -1
    
    def get_position_mm(self, max_width_mm: float = 85.0) -> float:
        """读取当前夹爪开口宽度（毫米）。
        返回 -1.0 表示读取失败。
        """
        pos = self.get_position()
        if pos >= 0:
            return self.position_to_width_mm(pos, max_width_mm)
        return -1.0
    
    def get_current(self) -> int:
        """读取当前电流（0-255，间接反映夹持力）。
        返回 -1 表示读取失败。
        电流越大，说明夹持力越大或遇到阻力越大。
        """
        resp = self.read_status(regs=3, verbose=False)
        if len(resp) >= 8:
            # 状态帧格式：[id, 0x03, byteCount, Byte0, Byte1, Byte2, Byte3, Byte4, Byte5, ...]
            # Byte4 是电流 gCU (current)
            gCU = resp[7]  # 索引 7 = Byte4
            return gCU
        return -1
    
    def get_object_status(self) -> int:
        """读取物体检测状态 gOBJ（0-3）。
        返回值含义：
          0: 手指运动中（未到位）
          1: 手指已停止，外部触发（抓到物体或碰到障碍）
          2: 手指已停止，内部触发（到达请求位置）
          3: 手指在外部或内部请求位置（已到位）
        返回 -1 表示读取失败。
        """
        resp = self.read_status(regs=3, verbose=False)
        if len(resp) >= 4:
            # gOBJ 在 Byte0 的 bit6-7
            gOBJ = (resp[3] >> 6) & 0x03
            return gOBJ
        return -1
    
    def check_grasp_success(self, timeout: float = 3.0, current_threshold: int = 20) -> dict:
        """检查是否成功抓取物体（带超时和电流阈值判断）。
        
        Args:
            timeout: 等待超时时间（秒）
            current_threshold: 电流阈值，超过此值认为接触到物体
        
        Returns:
            dict: {
                'success': bool,        # 是否抓取成功
                'gOBJ': int,            # 物体检测状态 0-3
                'position': int,        # 最终位置 0-255
                'current': int,         # 最终电流 0-255
                'width_mm': float,      # 最终开口宽度（毫米）
                'message': str          # 状态描述
            }
        """
        start_time = time.time()
        result = {
            'success': False,
            'gOBJ': -1,
            'position': -1,
            'current': -1,
            'width_mm': -1.0,
            'message': '检测中...'
        }
        
        while time.time() - start_time < timeout:
            gOBJ = self.get_object_status()
            current = self.get_current()
            position = self.get_position()
            
            if gOBJ == 1:
                # 外部触发停止（抓到物体）
                result['success'] = True
                result['gOBJ'] = gOBJ
                result['position'] = position
                result['current'] = current
                result['width_mm'] = self.position_to_width_mm(position)
                result['message'] = f'成功抓取！电流={current}, 位置={position}'
                return result
            elif gOBJ == 2:
                # 内部触发停止（到达目标位置但未抓到物体）
                if current >= current_threshold:
                    result['success'] = True
                    result['message'] = f'抓取成功（位置到位+电流达标）电流={current}'
                else:
                    result['success'] = False
                    result['message'] = f'未抓到物体（到位但电流低）电流={current}'
                result['gOBJ'] = gOBJ
                result['position'] = position
                result['current'] = current
                result['width_mm'] = self.position_to_width_mm(position)
                return result
            elif gOBJ == 3:
                # 已到位
                result['gOBJ'] = gOBJ
                result['position'] = position
                result['current'] = current
                result['width_mm'] = self.position_to_width_mm(position)
                if current >= current_threshold:
                    result['success'] = True
                    result['message'] = f'抓取成功（电流={current}）'
                else:
                    result['success'] = False
                    result['message'] = f'可能未抓到（电流={current}较低）'
                return result
            
            time.sleep(0.1)
        
        # 超时
        result['gOBJ'] = self.get_object_status()
        result['position'] = self.get_position()
        result['current'] = self.get_current()
        result['width_mm'] = self.position_to_width_mm(result['position'])
        result['message'] = f'超时：gOBJ={result["gOBJ"]}, 电流={result["current"]}'
        return result

    def _goto(self, rPR: int, rSP: int, rFR: int):
        """分步触发运动：先 rGTO=0（ctrl=0x01）装载参数，再 rGTO=1（ctrl=0x09）执行。"""
        # 预置参数，保持激活不动（rACT=1, rGTO=0 -> ctrl=0x01）
        self._send_command([0x01, 0x00, rPR, rSP, rFR, 0x00], wait_s=0.05)
        time.sleep(0.1)
        # 触发运动（rACT=1, rGTO=1 -> ctrl=0x09）
        self._send_command([0x09, 0x00, rPR, rSP, rFR, 0x00])
    
    def open_gripper(self, speed: int = 0xFF, force: int = 0xFF, wait: float = 2.0):
        """打开夹爪。
        speed/force 范围 0~255，越大越快/越大力。
        wait: 等待夹爪完全打开的时间（秒）
        """
        rSP = self._clamp_byte(speed)
        rFR = self._clamp_byte(force)
        # 使用分步触发，rPR=0（完全打开）
        self._goto(0x00, rSP, rFR)
        time.sleep(wait)  # 等待足够时间确保完全打开
    
    def close_gripper(self, speed: int = 0xFF, force: int = 0xFF):
        """关闭夹爪。"""
        rSP = self._clamp_byte(speed)
        rFR = self._clamp_byte(force)
        # 使用分步触发，rPR=255（完全闭合）
        self._goto(0xFF, rSP, rFR)
        time.sleep(1)
    
    def set_position(self, position: int, speed: int = 0xFF, force: int = 0xFF):
        """设置夹爪位置 (0-255, 0=完全打开, 255=完全关闭)。可指定速度/力度。"""
        rPR = self._clamp_byte(position)
        rSP = self._clamp_byte(speed)
        rFR = self._clamp_byte(force)
        self._goto(rPR, rSP, rFR)
        time.sleep(1)

    # =============== 便捷方法（宽度/角度换算） ===============
    @staticmethod
    def position_to_width_mm(position: int, max_width_mm: float = 85.0) -> float:
        """将 rPR(0-255) 线性映射为张开宽度（毫米）。0 -> 最大宽度，255 -> 0。
        注意：Robotiq 是平行夹爪，没有“角度”，是“开口宽度”。
        """
        p = max(0, min(255, int(position)))
        return (1.0 - p / 255.0) * max_width_mm

    @staticmethod
    def width_mm_to_position(width_mm: float, max_width_mm: float = 85.0) -> int:
        """将目标开口宽度（毫米）映射为 rPR(0-255)。线性近似。"""
        w = max(0.0, min(max_width_mm, float(width_mm)))
        pos = int(round((1.0 - w / max_width_mm) * 255.0))
        return max(0, min(255, pos))

    def set_width(self, width_mm: float, speed: int = 0xFF, force: int = 0xFF):
        """按开口宽度（毫米）设定位置，内部转换为 rPR。"""
        rPR = self.width_mm_to_position(width_mm)
        self.set_position(rPR, speed=speed, force=force)
    
    def monitor_status(self, threshold_mm: float = 50.0, interval: float = 0.5):
        """持续监控夹爪状态并输出。
        
        Args:
            threshold_mm: 开口宽度阈值（毫米），<= threshold_mm 输出 1（闭合），> threshold_mm 输出 0（打开）
            interval: 监控间隔时间（秒）
        
        按 Ctrl+C 停止监控。
        """
        print(f"\n=== 夹爪状态监控 ===")
        print(f"阈值设定: 开口宽度 <= {threshold_mm} mm 为闭合状态(1)，> {threshold_mm} mm 为打开状态(0)")
        print(f"监控间隔: {interval} 秒")
        print("按 Ctrl+C 停止监控\n")
        
        try:
            while True:
                width = self.get_position_mm()
                
                if width < 0:
                    print("读取失败")
                    time.sleep(interval)
                    continue
                
                # 判断状态：<= 50mm 为闭合(1)，> 50mm 为打开(0)
                status = 1 if width <= threshold_mm else 0
                
                # 输出格式：状态值 | 开口宽度
                print(f"{status} | 开口宽度: {width:.1f} mm")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止")
    
    def monitor_status_background(self, threshold_mm: float = 50.0, interval: float = 0.0, stop_event=None):
        """后台监控夹爪状态并输出（用于线程）。
        
        Args:
            threshold_mm: 开口宽度阈值（毫米），<= threshold_mm 输出 1（闭合），> threshold_mm 输出 0（释放）
            interval: 监控间隔时间（秒），0 表示尽可能快
            stop_event: threading.Event 对象，用于停止监控
        """
        try:
            last_time = time.time()
            target_interval = 0.1  # 目标 10Hz
            
            while not (stop_event and stop_event.is_set()):
                loop_start = time.time()
                
                width = self.get_position_mm()
                
                if width < 0:
                    time.sleep(0.01)
                    continue
                
                # 判断状态：<= 50mm 为闭合(1)，> 50mm 为释放(0)
                status = 1 if width <= threshold_mm else 0
                status_text = "闭合" if status == 1 else "释放"
                
                # 计算实际刷新率
                current_time = time.time()
                actual_hz = 1.0 / (current_time - last_time) if (current_time - last_time) > 0 else 0
                last_time = current_time
                
                # 输出格式：状态值|状态文字 [实际Hz]
                print(f"{status}|{status_text} [{actual_hz:.1f}Hz]")
                
                # 动态调整等待时间以达到目标刷新率
                elapsed = time.time() - loop_start
                sleep_time = max(0, target_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except Exception as e:
            print(f"\n监控线程错误: {e}")
    
    def close(self):
        """关闭串口连接"""
        self.ser.close()
        print("串口已关闭")

# 使用示例
if __name__ == "__main__":
    import sys
    import msvcrt  # Windows 下的键盘检测
    
    # 修改为你的实际串口号，debug=False 关闭 TX/RX 输出
    gripper = Robotiq2F85(port='COM5', debug=False)
    
    # 激活夹爪
    gripper.activate()
    
    # 初始状态：释放
    current_status = 0  # 0=释放, 1=闭合
    
    print("\n=== 夹爪控制 ===")
    print("按键说明:")
    print("  Q - 闭合夹爪（夹取物体）")
    print("  E - 打开夹爪（松开物体）")
    print("  X - 退出程序")
    print("\n开始以 10Hz 输出状态...\n")
    
    try:
        while True:
            # 检查是否有按键
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8').upper()
                
                if key == 'Q':
                    current_status = 1
                    #gripper.close_gripper(speed=100, force=170)
                    gripper.open_gripper(speed=255, force=200, wait=3.0)
                elif key == 'E':
                    current_status = 0
                    #gripper.open_gripper(speed=255, force=200, wait=3.0)
                    gripper.close_gripper(speed=100, force=170)
                elif key == 'X':
                    print("\n>>> 退出程序...")
                    break
            
            # 以 10Hz 输出当前状态
            status_text = "闭合" if current_status == 1 else "释放"
            print(f"{current_status}|{status_text}")
            
            # 等待 0.1 秒 (10Hz)
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n\n程序被中断")
    
    # 关闭连接
    gripper.close()
    print("串口已关闭，程序结束")