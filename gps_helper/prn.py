import json
import os

prn_json = open(os.path.join(os.path.dirname(__file__), 'prn_info.json')).read()
prn_info = json.loads(prn_json)