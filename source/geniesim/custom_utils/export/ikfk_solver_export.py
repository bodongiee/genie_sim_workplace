import contextlib
import os
import sys
import tempfile

# =============================================================================
# ===== C++ stdout capture (reused from MuJoCo version) =====
# =============================================================================
@contextlib.contextmanager
def capture_c_stdout():
    fd = sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else 1
    with tempfile.TemporaryFile(mode="w+b") as tfile:
        old_stdout = os.dup(fd)
        try:
            sys.stdout.flush()
            os.dup2(tfile.fileno(), fd)
            yield
        finally:
            os.dup2(old_stdout, fd)
            os.close(old_stdout)
            tfile.seek(0)
            output = tfile.read().decode("utf-8", errors="replace")
            if output.strip():
                translations = {
                    "========== 初始化 Solver ==========": "========== Initialize Solver ==========",
                    "机器人部件:": "Robot Part:",
                    "正在加载配置文件:": "Loading config file:",
                    "配置文件加载成功": "Config file loaded successfully",
                    "正在初始化RobotWrapper...": "Initializing RobotWrapper...",
                    "成功加载完整模型，路径为:": "Successfully loaded full model, path:",
                    "成功加载配置文件:": "Successfully loaded config file:",
                    "目标维度设置为:": "Target dimension set to:",
                    "已加载右臂关节配置": "Right arm joint config loaded",
                    "已加载左臂关节配置": "Left arm joint config loaded",
                    "========== 简化模型信息 ==========": "========== Simplified Model Info ==========",
                    "自由度:": "DOF:",
                    "----- 可动关节列表 -----": "----- Active Joint List -----",
                    "成功加载简化模型": "Successfully loaded simplified model",
                    "已设置关节限位：": "Joint limits set:",
                    "位置下限:": "Pos Lower Limit:",
                    "位置上限:": "Pos Upper Limit:",
                    "速度限制:": "Vel Limit:",
                    "力矩限制:": "Torque Limit:",
                    "RobotWrapper初始化完成": "RobotWrapper initialization complete",
                    "状态维度:": "State dim:",
                    "目标维度:": "Target dim:",
                    "Variables初始化完成": "Variables initialization complete",
                    "初始化完成": "Initialization complete",
                }
                for cn, en in translations.items():
                    output = output.replace(cn, en)
                print('\033[34m' + f" [C++ IK Solver]:\n{output}" + '\033[0m')
