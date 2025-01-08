import os
import configparser
from .dataset import ROOT_PATH

CONFIG_PATH = os.path.join(ROOT_PATH, "config.ini")

conf_parser = configparser.RawConfigParser()
conf_parser.read(CONFIG_PATH)

DB_CONFIG = conf_parser._sections["db"]
