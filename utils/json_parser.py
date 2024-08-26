import json

from typing import Dict, Any

from utils.config import Config


def parse_json(path: str) -> Config:
    with open(path, "r") as file:
        data: Dict[str, Any] = json.load(file)
    config: Config = Config(**data)

    return config
