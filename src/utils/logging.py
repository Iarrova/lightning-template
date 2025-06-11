import logging
from typing import List

import structlog
from structlog.types import Processor


def setup_logger(name: str, log_level: int = logging.INFO) -> structlog.BoundLogger:
    """
    Set up and configure a structlog logger.

    Args:
        log_level: The logging level for the console handler. Default is logging.INFO.

    Returns:
        A configured structlog BoundLogger instance
    """
    logger = logging.getLogger(name)
    # Avoid adding duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return structlog.get_logger(name)

    logger.setLevel(log_level)
    logger.propagate = False

    shared_processors: List[Processor] = [
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.StackInfoRenderer(),
    ]

    structlog.configure(
        logger_factory=structlog.stdlib.LoggerFactory(),
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return structlog.get_logger(name)
