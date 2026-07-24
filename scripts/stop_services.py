#!/usr/bin/env python3
"""
听书知识库 - 服务停止脚本
用法：python scripts/stop_services.py [--all] [--import] [--query] [--dashboard]
"""
import os
import sys
import subprocess
import signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def is_wsl():
    """检测是否在 WSL 环境中"""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False

def find_process_by_port(port: int) -> list[int]:
    """通过端口查找进程 PID"""
    pids = []
    try:
        if is_wsl():
            # WSL 环境
            result = subprocess.run(
                ['bash', '-c', f'netstat -tlnp 2>/dev/null | grep :{port} | awk \'{{print $NF}}\' | grep -o \'[0-9]*\''],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line and line.isdigit():
                    pid = int(line)
                    pids.append(pid)
        else:
            # Windows 环境
            result = subprocess.run(
                ['powershell', '-Command', f'Get-NetTCPConnection -LocalPort {port} | Select-Object OwningProcess'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and line.isdigit():
                    pids.append(int(line))
    except Exception as e:
        print(f"⚠️  查找端口 {port} 进程失败: {e}")
    return pids

def find_process_by_name(name_pattern: str) -> list[int]:
    """通过进程名查找 PID"""
    pids = []
    try:
        if is_wsl():
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                if name_pattern in line and 'python' in line:
                    parts = line.split()
                    if parts and parts[1].isdigit():
                        pids.append(int(parts[1]))
        else:
            result = subprocess.run(
                ['powershell', '-Command', f'Get-WmiObject Win32_Process | Where-Object {{ $_.Name -like "*{name_pattern}*" }} | Select-Object ProcessId'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.split('\n'):
                line = line.strip()
                if line and line.isdigit():
                    pids.append(int(line))
    except Exception as e:
        print(f"⚠️  查找进程 {name_pattern} 失败: {e}")
    return pids

def kill_process(pid: int, name: str, timeout: int = 5) -> bool:
    """终止进程，先尝试 SIGTERM，等待后未退出则用 SIGKILL"""
    try:
        # 先尝试优雅终止
        if is_wsl():
            os.kill(pid, signal.SIGTERM)
        else:
            subprocess.run(['taskkill', '/PID', str(pid), '/T'], capture_output=True)
        
        # 等待进程终止
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # 检查进程是否还存在
                if is_wsl():
                    os.kill(pid, 0)  # 不发送信号，只检查是否存在
                else:
                    result = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], 
                                          capture_output=True, text=True)
                    if str(pid) not in result.stdout:
                        return True
                time.sleep(0.2)
            except ProcessLookupError:
                return True  # 进程已不存在
            except Exception:
                pass
        
        # 如果还没退出，强制终止
        print(f"   ⚠️  进程未响应，强制终止...")
        if is_wsl():
            os.kill(pid, signal.SIGKILL)
        else:
            subprocess.run(['taskkill', '/F', '/PID', str(pid)], capture_output=True)
        time.sleep(0.5)
        return True
    except ProcessLookupError:
        print(f"   ℹ️  进程 {pid} 已不存在")
        return True
    except Exception as e:
        print(f"   ❌ 停止 {name} 失败: {e}")
        return False

def stop_service(service_name: str, port: int, process_pattern: str = None):
    """停止指定服务"""
    print(f"\n📋 检查 {service_name} (端口 {port})...")
    
    # 通过端口查找
    pids = find_process_by_port(port)
    
    # 如果端口没找到，通过进程名查找
    if not pids and process_pattern:
        pids = find_process_by_name(process_pattern)
    
    if not pids:
        print(f"   ℹ️  {service_name} 未运行")
        return
    
    print(f"   找到 {len(pids)} 个进程: {pids}")
    for pid in pids:
        kill_process(pid, service_name)

def main():
    print("=" * 50)
    print("🛑  听书知识库 - 服务停止工具")
    print("=" * 50)
    
    # 解析参数
    args = sys.argv[1:]
    stop_all = not args or '--all' in args
    stop_import = '--import' in args or stop_all
    stop_query = '--query' in args or stop_all
    stop_dashboard = '--dashboard' in args or stop_all
    
    if '--help' in args or '-h' in args:
        print("\n用法：")
        print("  python scripts/stop_services.py [选项]")
        print("\n选项：")
        print("  --all        停止所有服务（默认）")
        print("  --import     仅停止导入服务 (8001)")
        print("  --query      仅停止查询服务 (8002)")
        print("  --dashboard  仅停止控制面板 (8080)")
        print("  --help, -h   显示此帮助")
        return
    
    stopped_any = False
    
    # 停止导入服务
    if stop_import:
        stop_service("导入服务", 8001, "import_server")
        stopped_any = True
    
    # 停止查询服务
    if stop_query:
        stop_service("查询服务", 8002, "query_server")
        stopped_any = True
    
    # 停止控制面板
    if stop_dashboard:
        stop_service("控制面板", 8080, "control_server")
        stopped_any = True
    
    print("\n" + "=" * 50)
    if stopped_any:
        print("✨ 服务停止完成")
    else:
        print("⚠️  未指定要停止的服务")
    print("=" * 50)

if __name__ == "__main__":
    main()
