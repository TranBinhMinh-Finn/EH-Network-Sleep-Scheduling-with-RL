from datetime import datetime
import logging

logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
file_name = f"logs/{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
file_handler = logging.FileHandler(file_name)
logger.addHandler(file_handler)

LOG_TYPE = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40
}


def _get_log_type(_type):
    for key in LOG_TYPE.keys():
        if LOG_TYPE[key] == _type:
            return key
    return "INFO"


def _log(_type, module_name, *args):
    logger.log(_type, f"[{module_name}]: {' '.join(map(str, args))}")
    print(f"[{_get_log_type(_type)}] [{module_name}]: {' '.join(map(str, args))}")


def debug(module_name, *args):
    logger.debug((f"[DEBUG] [{module_name}]: {' '.join(map(str, args))}"))


def info(module_name, *args):
    _log(LOG_TYPE["INFO"], module_name, *args)


def warning(module_name, *args):
    _log(LOG_TYPE["WARNING"], module_name, *args)


def error(module_name, *args):
    _log(LOG_TYPE["ERROR"], module_name, *args)


def delete_log():
    file_handler.close()
    logger.removeHandler(file_handler)
    import os
    os.remove(file_name)


__all__ = ["debug", "info", "warning", "error", "delete_log"]