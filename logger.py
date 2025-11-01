'''
这是一个学习日志记录功能的示例代码，展示了如何使用Python的logging模块来创建一个简单的日志记录器。
给出了一个统一的logging实例
'''
import logging
from typing import Any


class Logger(object):
    def __new__(cls, loggername, loggerfilename, *args, **kwargs):
        logger = logging.getLogger(loggername)
        logger.setLevel(logging.DEBUG)

        # 日志输出渠道
        console_log = logging.StreamHandler()
        console_log.setLevel(logging.INFO)
        file_log = logging.FileHandler(
            filename=loggerfilename, mode='w', encoding='utf-8')
        file_log.setLevel(logging.DEBUG)
        # 日志输出格式
        formatter = "%(asctime)s - %(levelname)s -  %(pathname)s -  %(lineno)s -%(funcName)s --%(message)s"
        formatter = logging.Formatter(formatter)
        console_log.setFormatter(formatter)
        file_log.setFormatter(formatter)
        # 添加日志渠道
        logger.addHandler(console_log)
        logger.addHandler(file_log)

        return logger

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)


if __name__ == "__main__":
    logger = Logger("mylogger", "app.log")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
