import json
from logging import Logger
import os
from urllib import response
import os.path as osp
import torch
import torch.distributed as dist
import csv

# sys.path.append
from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint

from aws_utils import aws_util
from enap_utils import make_artifacts, get_modelconfig, metric_logging, dataset_logging, Dict2Class

default_args = {"eval_interval":20, "eval_final_only":False,
                 "heavy_eval_range":50, "check_images":True,
                 "check_labels":True, "name":"exp", "dist_url":"env://",
                 "gpu_count":0, "local_rank":-1, "resume":False, "workers":8,
                 "rank":-1, "world_size":1, "output_dir":"./artifacts",
                 "save_dir":"./runs", "data_path": "../data.yaml"
                }

WORK_DIR = "./"
torch.backends.cudnn.benchmark = True

# Load train_params.json config
train_params_path = os.environ.get("TRAIN_PARAMS_LOC","./train_params.json")
args = json.loads(open(train_params_path, 'r').read())
args = args['config_json']

# extend the loaded json to default values
args.update(default_args)


# select and load model config
args["conf_file"] = get_modelconfig(WORK_DIR, args['modelname'])
cfg = Config.fromfile(args["conf_file"])


device = select_device(args['device'])

# loading the args dict into class
args_class = Dict2Class(args) 

SESSION_ID = args["sessionid"]
UPDATE_DATASET_URL = args["UPDATE_DATASET_URL"]
UPDATE_STATUS_URL = args["UPDATE_STATUS_URL"]

# Calling Trainer Class for model and dataset loading-----
trainer = Trainer(args_class, cfg, device)

print("-------------------------writing dataset.csv before training starts-------------------------")
dataset_csv = os.path.join(WORK_DIR, "dataset.csv")
dataset_response = dataset_logging(trainer, UPDATE_DATASET_URL, SESSION_ID, dataset_csv)
print(dataset_response)
print("-----------------------------dataset.csv pused to the endpoint------------------------------")

# Training and Metrics
with open('./metrics.csv', 'w') as f:
    w = csv.writer(f)
    losses = []
    columns = ["epoch", "loss", "IOU_loss", "l1_loss", "obj_loss", "cls_loss", "bbox_mAP_0.5", "bbox_mAP_0.50_0.95"]
    w.writerow(columns)
    losses.append(columns)

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
            w.writerow(loss)
            losses.append(loss)
            
            metric_logging(UPDATE_STATUS_URL, SESSION_ID, "./metrics.csv", trainer.epoch)
            
    except Exception as _:
        LOGGER.error('ERROR in training loop or eval/save model.')
        raise
    finally:
        trainer.train_after_loop()

make_artifacts(args_class)

# try:
#     aws_U = aws_util()
#     aws_U.upload_file(
#         WORK_DIR+'.zip',
#         'enap-train-data',
#         SESSION_ID+"/artifacts.zip"
#         )
#     os.system('rm {}.zip'.format(WORK_DIR))
# except Exception as e:
#     print(e)
