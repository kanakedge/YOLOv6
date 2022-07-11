import argparse
import json
from logging import Logger
import os
from typing import DefaultDict
import yaml
import os.path as osp
from pathlib import Path
import torch
import torch.distributed as dist
import sys, shutil
import pprint
import csv

# sys.path.append
from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint

def make_artifacts(config):
    os.makedirs('./artifacts', exist_ok=True)
    shutil.copy(config.conf_file, './artifacts')
    shutil.copy('./dataset.csv','./artifacts')
    shutil.copy('./metrics.csv','./artifacts')
    shutil.copy('./artifacts/weights/best_ckpt.pt','./artifacts/final_model.pt')
    
    shutil.make_archive('artifacts','zip','./artifacts')
    shutil.copy('./artifacts.zip','./artifacts')


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])    

default_args = {"eval_interval":20, "eval_final_only":False,
                 "heavy_eval_range":50, "check_images":True,
                 "check_labels":True, "name":"exp", "dist_url":"env://",
                 "gpu_count":0, "local_rank":-1, "resume":False, "workers":8,
                 "rank":-1, "world_size":1, "save_dir":'./artifacts',
                 "output_dir":'./artifacts'}

torch.backends.cudnn.benchmark = True

train_params_path = os.environ.get("TRAIN_PARAMS_LOC","./train_params.json")
args = json.loads(open(train_params_path, 'r').read())
args.update(default_args)
cfg = Config.fromfile(args["conf_file"])
device = select_device(args['device'])
args_class = Dict2Class(args) # loading the args dict into class


# Calling Trainer Class -----
trainer = Trainer(args_class, cfg, device)

# ----- writing dataset.csv before training starts ------
dataset_dict = {"train_set_size":trainer.len_train, "val_set_size":trainer.len_valid,
                 "total_classes":trainer.data_dict['nc']}

with open('dataset.csv', 'w') as f:
    w = csv.DictWriter(f, dataset_dict.keys())
    w.writeheader()
    w.writerow(dataset_dict)
print("---- dataset.csv written ----")
# ---- dataset.csv written ----

losses = []
losses.append(["epoch", "loss", "IOU_loss", "l1_loss", "obj_loss", "cls_loss", "bbox_mAP_0.5", "bbox_mAP_0.50_0.95"])

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
        
except Exception as _:
    LOGGER.error('ERROR in training loop or eval/save model.')
    raise
finally:
    trainer.train_after_loop()

print(len(losses))
with open('./metrics.csv', 'w') as f:
    w = csv.writer(f)
    w.writerows(losses)


make_artifacts(args_class)
