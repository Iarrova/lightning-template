import json
from typing import Any, Dict

from config.config import Config


class ConfigParser:
    @staticmethod
    def parse_json(path: str) -> Config:
        try:
            with open(path, "r") as file:
                data: Dict[str, Any] = json.load(file)
            config: Config = Config(**data)

            return config

        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise ValueError(f"Error parsing configuration file: {str(e)}")
