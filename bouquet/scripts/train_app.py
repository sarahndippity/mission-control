import argparse
import logging
import os
import yaml

from src.inference_model import XGBoostModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')

if __name__ == "__main__":
    """
    Author: Sarah Xie
    Source:
    
    This script is used to train any XGBoost-type model.
    To run the script, use
    >>> python3 train_app.py -c <path_to_config> -m <model_name>
    
    To add a new model type to this script, add the newly-created class to
    model_store = {"model_name": ModelClassName()}
    """
    parser = argparse.ArgumentParser(description="Script to train any XGBoost-type model.")

    parser.add_argument("-c", "--config", type=str, help="Path to config files",
                        default="configs/xgb_configs.yaml")

    path_to_config = parser.parse_args().config
    with open(path_to_config, 'r') as f:
        configs = yaml.safe_load(f)

    model_kwargs = {
        "seed": configs["random_seed"],
        "learning_rate": configs["xgb"]["learning_rate"],
        "gamma": configs["xgb"]["gamma"],
        "max_depth": configs["xgb"]["max_depth"],
        "subsample": configs["xgb"]["subsample"],
        "colsample_bytree": configs["xgb"]["colsample_bytree"],
        "lambda": configs["xgb"]["lambda"],
        "alpha": configs["xgb"]["alpha"],
        "scale_pos_weight": configs["xgb"]["scale_pos_weight"],
        "max_leaves": configs["xgb"]["max_leaves"],
        "objective": configs["xgb"]["objective"],
        "multi_strategy": configs["xgb"]["multi_strategy"],
        "eval_metric": configs["xgb"]["eval_metric"]
    }

    # add any new model types to model_store dict below
    model_store = {
        "xgb": XGBoostModel(data_path=configs["paths"]["data_path"],
                            target=configs["target_variable"],
                            xgb_kwargs=model_kwargs,
                            max_features=configs["max_features"],
                            save_path=configs["paths"]["save_path"])
    }

    # instantiate & call model
    model_name = parser.parse_args().model
    model = model_store[model_name]
    model.run()
