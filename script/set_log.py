import os
import sys
import time
import argparse
import logging
from datetime import datetime


def setup_logging(output_path: str, log_level=logging.INFO):
    """
    设置日志记录，同时输出到控制台和文件
    
    Args:
        output_path: 输出目录路径，日志文件将保存在此目录下
        log_level: 日志级别，默认为 INFO
    
    Returns:
        logger: 日志记录器
        log_file: 日志文件路径
    """
    # 创建日志目录
    log_dir = os.path.join(output_path, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # 生成日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_eval_{timestamp}.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            # 文件处理器 - 记录所有级别的日志
            logging.FileHandler(log_file, encoding='utf-8'),
            # 控制台处理器 - 只显示 INFO 及以上级别
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # 强制重新配置，避免重复配置问题
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件已创建: {log_file}")
    logger.info(f"日志级别: {logging.getLevelName(log_level)}")
    
    return logger, log_file