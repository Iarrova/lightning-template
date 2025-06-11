import json
import os
from typing import Any, Dict

from src.config.config import Config
from src.exceptions import ConfigurationError
from src.utils.logging import setup_logger

logger = setup_logger(__name__)


class ConfigParser:
    @staticmethod
    def parse_json(path: str) -> Config:
        try:
            if not os.path.exists(path):
                raise ConfigurationError(f"Configuration file not found: {path}")

            with open(path, "r") as file:
                try:
                    data: Dict[str, Any] = json.load(file)
                except json.JSONDecodeError as e:
                    raise ConfigurationError(f"Invalid JSON in configuration file: {str(e)}")

            logger.debug(f"Loaded configuration from {path}")

            try:
                config: Config = Config(**data)
            except Exception as e:
                raise ConfigurationError(f"Error validating configuration: {str(e)}")

            logger.debug("Configuration validated successfully")
            return config

        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(f"Error parsing configuration file: {str(e)}")
            raise
