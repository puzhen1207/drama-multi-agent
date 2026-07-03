"""结构化日志配置。"""
from __future__ import annotations

import logging
import sys

from loguru import logger as _loguru_logger

from .config import settings


class InterceptHandler(logging.Handler):
    """把标准 logging 的消息转发到 loguru。"""

    def emit(self, record):
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """初始化全局日志。"""
    _loguru_logger.remove()
    _loguru_logger.add(
        sys.stdout,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        enqueue=True,
    )
    for name in ("httpx", "urllib3", "faiss", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.basicConfig(handlers=[InterceptHandler()], level=settings.log_level, force=True)


def get_logger(name: str = "drama_agent"):
    return _loguru_logger.bind(module=name)
