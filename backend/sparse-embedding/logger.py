import logging
import sys
from config import GlobalConfig
import time

def setup_logger(name: str = "LegalAgent", log_filename: str = "app.log", level=logging.DEBUG):
    """ 初始化日志记录器
    :param name: Logger 名称
    :param log_filename: 日志文件名 (将保存在 logs/ 目录下)
    :param level: 日志级别
    """
    # 1. 获取日志目录
    log_path = GlobalConfig.LOG_DIR / log_filename
    
    # 2. 创建 Logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 3. 清理已存在的 handlers，确保每次重新创建
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # 4. 确保每次运行都是新的log
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = GlobalConfig.LOG_DIR / f"{log_filename.rsplit('.', 1)[0]}_{timestamp}.{log_filename.rsplit('.', 1)[1] if '.' in log_filename else 'log'}"

    # 5. 定义格式
    # 格式：时间 | 级别 | 模块名 | 消息
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 6. 控制台 Handler (输出到终端)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # 7. 文件 Handler (输出到文件)
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)
    
    return logger

# 创建一个默认的 logger 实例供直接导入使用
logger = setup_logger(name="SparseEmbedding", log_filename="SparseEmbedding.log")