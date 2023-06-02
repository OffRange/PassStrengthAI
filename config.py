import json

CONFIG = {}


def init():
    from sys import argv
    config_path = 'config.json'
    if "--config" in argv:
        index = argv.index("--config")
        if index + 1 < len(argv):
            config_path = argv[index + 1]

    global CONFIG
    CONFIG = json.load(open(config_path))


init()
