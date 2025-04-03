"""
This loads configurations for the planner from a YAML file in config directory.
"""

import yaml
import os

def load_planner_params(planner_name, config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "../../config/planner_params.yaml")
    with open(config_path, "r") as f:
        all_params = yaml.safe_load(f)
    return all_params.get(planner_name, {})
