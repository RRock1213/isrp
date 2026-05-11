"""安全打印工具 — 解决 Windows 控制台 GBK 编码问题"""
import sys


def safe_print(text: str, **kwargs):
    """安全打印，处理 Windows GBK 编码问题"""
    try:
        print(text, **kwargs)
    except UnicodeEncodeError:
        if sys.platform == 'win32':
            encoded = text.encode('gbk', errors='replace').decode('gbk')
            print(encoded, **kwargs)
        else:
            encoded = text.encode('utf-8', errors='replace').decode('utf-8')
            print(encoded, **kwargs)


def safe_format(text: str, max_len: int = 50) -> str:
    """安全格式化文本，处理编码问题"""
    if len(text) > max_len:
        text = text[:max_len] + "..."
    try:
        if sys.platform == 'win32':
            text.encode('gbk')
        return text
    except UnicodeEncodeError:
        if sys.platform == 'win32':
            return text.encode('gbk', errors='replace').decode('gbk')
        return text


def setup_windows_console():
    """设置 Windows 控制台为 UTF-8 模式"""
    if sys.platform != 'win32':
        return
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


setup_windows_console()
