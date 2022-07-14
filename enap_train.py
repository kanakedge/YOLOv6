from inspect import ArgSpec
import json
import logging
import os
from urllib import response
import os.path as osp
import torch
import torch.distributed as dist
import csv
import traceback
import requests

# sys.path.append
from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint

from aws_utils import aws_util
from enap_utils import make_artifacts, get_modelconfig, metric_logging, dataset_logging, Dict2Class

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

def append_metrics(metrics_csv, row, mode='a'):
        with open(metrics_csv, mode) as f:
            w = csv.writer(f)
            w.writerow(row)

default_args = {"eval_interval":20, "eval_final_only":False,
                 "heavy_eval_range":50, "check_images":True,
                 "check_labels":True, "name":"exp", "dist_url":"env://",
                 "gpu_count":0, "local_rank":-1, "resume":False, "workers":8,
                 "rank":-1, "world_size":1, "output_dir":"./artifacts",
                 "save_dir":"./runs", "data_path": "./data.yaml", "img_size": [640,640]
                }
print(default_args, type(default_args))

WORK_DIR = "./"
torch.backends.cudnn.benchmark = True

# Load train_params.json config
train_params_path = os.environ.get("TRAIN_PARAMS_LOC","./train_params.json")

args = json.loads(open(train_params_path, 'r').read())

try:
    args = args['config_json']
    if isinstance(args, str):
        args = json.loads(args)
    print(args) 
except Exception as _:
    logging.debug("Error while json.loads(args)")
    raise

# extend the loaded json to default values
args.update(default_args)
args['img_size'] = args.get('img-size', args['img_size'])
print(args.keys())

# select and load model config
args["conf_file"] = get_modelconfig(WORK_DIR, args['modelname'], args['resume'])
cfg = Config.fromfile(args["conf_file"])


device = select_device(args['device'])

# loading the args dict into class
args_class = Dict2Class(args) 

SESSION_ID = args["sessionid"]
UPDATE_DATASET_URL = args["UPDATE_DATASET_URL"]
UPDATE_STATUS_URL = args["UPDATE_STATUS_URL"]

def main():
    
    # Calling Trainer Class for model and dataset loading-----
    trainer = Trainer(args_class, cfg, device)

    print("-------------------------writing dataset.csv before training starts-------------------------")
    dataset_csv = os.path.join(WORK_DIR, "dataset.csv")
    dataset_response = dataset_logging(trainer, UPDATE_DATASET_URL, SESSION_ID, dataset_csv)
    print(dataset_response)
    print("-----------------------------dataset.csv pused to the endpoint------------------------------")

    # Training and Metrics
    columns = ["Epoch", "train_total_loss", "train_box_loss", "train_l1_loss", "train_obj_loss", "train_cls_loss", "mAP@.5", "mAP@.5:.95"]
    losses = []
    losses.append(columns)
    
    metrics_csv = "./metrics.csv"

    # with open('./metrics.csv', 'w') as f:
    #     w = csv.writer(f)
    #     #train_box_loss is IOU
    #     w.writerow(columns)
    
    append_metrics(metrics_csv, columns, 'w')

    try:
        trainer.train_before_loop()
        for trainer.epoch in range(trainer.start_epoch, trainer.max_epoch):
            trainer.train_in_loop()
            
            trainer_loss_items = trainer.loss_items
            trainer_loss_items = trainer_loss_items.cpu().numpy()
            total_loss = float(trainer.total_loss.detach().cpu().numpy())
            evaluation_results = trainer.evaluate_results
            
            loss = [trainer.epoch, total_loss]
            loss.extend(trainer_loss_items)
            loss.extend(evaluation_results)
            losses.append(loss)
            append_metrics(metrics_csv, loss)
            metric_response = metric_logging(UPDATE_STATUS_URL, SESSION_ID, "./metrics.csv", trainer.epoch)
            print(metric_response)
            
    except Exception as _:
        LOGGER.error('ERROR in training loop or eval/save model.')
        raise
    finally:
        trainer.train_after_loop()

    make_artifacts(args_class)

    try:
        artifacts_zip = os.path.join(args.get('output_dir'), "artifacts.zip")
        if not os.path.exists(artifacts_zip):
            logging.debug("artifacts.zip not found!")
        aws_U = aws_util()
        aws_U.upload_file(
            artifacts_zip,
            'enap-train-data',
            SESSION_ID+"/artifacts.zip"
            )
        os.system('rm {}.zip'.format(WORK_DIR))
    except Exception as e:
        print(e)



if __name__=="__main__":

    train_response_params = {"status": False, "status_code": 500, "sessionid": 0}
    train_response_params['sessionid'] = SESSION_ID
    train_response_params['modelname'] = args["modelname"]

    try :
        main()

        train_response_params["status_code"] = 200
        train_response_params["status"] = "Completed"

    except Exception as e :
        train_response_params["status_code"] = 523
        train_response_params["error"] = "Training Failed"
        train_response_params["status"] = "Failed"
        print("Error!!!=================")
        print(e)
        print(traceback.format_exc())
    response = requests.request("POST",UPDATE_STATUS_URL,headers={},json=train_response_params)
    print(response.content)
