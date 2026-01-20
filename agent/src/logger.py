import logging
import sys
from config import Config
import time
from pathlib import Path

def setup_logger(name: str = "LegalAgent", log_filename: str = "app.log", level=logging.DEBUG):
    """ 初始化日志记录器
    :param name: Logger 名称
    :param log_filename: 日志文件名 (将保存在 logs/ 目录下)
    :param level: 日志级别
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(Config.LOG_DIR)
    log_path = log_dir / f"{log_filename.rsplit('.', 1)[0]}_{timestamp}.{log_filename.rsplit('.', 1)[1] if '.' in log_filename else 'log'}"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(name="DenseEmbedding", log_filename="DenseEmbedding.log")