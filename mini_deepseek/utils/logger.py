import os
import sys
from pathlib import Path

from loguru import logger as _loguru_logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"

LOG_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level}</level> | <cyan>{file.path}</cyan>:<cyan>{line}</cyan> | <magenta>{function}</magenta> | <level>{message}</level>"
FILE_LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {file.path}:{line} | {function} | {message}"


def setup_logger():
    """Configure the shared project logger and return it."""
    # 移除 loguru 默认 handler，避免重复输出。
    _loguru_logger.remove()
    rank = int(os.environ.get("RANK", 0))

    # 仅主进程输出到控制台和统一日志文件，避免多卡时日志重复刷屏。
    if rank == 0:
        _loguru_logger.add(sys.stdout, format=LOG_FORMAT, level="DEBUG", colorize=True, enqueue=True)

        # 文件 sink 用纯文本格式，方便 grep / tail / 持久化排查。
        _loguru_logger.add(
            LOG_FILE,
            format=FILE_LOG_FORMAT,
            level="DEBUG",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,
            diagnose=True,
        )

    return _loguru_logger


logger = setup_logger()

# 示例
if __name__ == "__main__":
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
