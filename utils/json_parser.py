import json
from utils.config import Config


def parse_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    config = Config(**data)

    return config
