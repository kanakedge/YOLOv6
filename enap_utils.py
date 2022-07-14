import os, shutil
import requests
import csv

def make_artifacts(config):
    os.makedirs('./artifacts', exist_ok=True)
    shutil.copy(config.conf_file, './artifacts')
    shutil.move('./dataset.csv','./artifacts')
    shutil.move('./metrics.csv','./artifacts')
    shutil.copy('./runs/weights/best_ckpt.pt','./artifacts/final_model.pt')

    shutil.make_archive('artifacts','zip','./artifacts')
    shutil.move('./artifacts.zip','./artifacts')


def get_modelconfig(WORK_DIR, modelname, resume=False):
    config_folder = f"{WORK_DIR}/configs"
    config_dict = {
        "yolov6n": ["yolov6n.py", "yolov6n_finetune.py"],
        "yolov6s": ["yolov6s.py", "yolov6s_finetune.py"],
        "yolov6_tiny": ["yolov6_tiny.py", "yolov6_tiny_finetune.py"]
    }
    if resume:
        return os.path.join(config_folder, config_dict[modelname][1])
    return os.path.join(config_folder, config_dict[modelname][0])

def metric_logging(UPDATE_STATUS_URL, SESSION_ID, metrics_file, epoch):
    update_post_params = dict()
    update_post_params["status_code"] = 200
    update_post_params["error"] = ""
    update_post_params["status"] = "In Progress"
    update_post_params["epochs"] = epoch
    update_post_params["sessionid"] = SESSION_ID
    metrics_csv_file = [('metrics.csv', open(metrics_file,'rb'))]
    headers = {}
    response = requests.request("POST", UPDATE_STATUS_URL, headers=headers, data=update_post_params , files=metrics_csv_file)
    return response.content

def dataset_logging(trainer, UPDATE_DATASET_URL, SESSION_ID, dataset_csv):
    dataset_dict = {"train_set_size":trainer.len_train, "val_set_size":trainer.len_valid,
                    "total_classes":trainer.data_dict['nc']}

    with open(dataset_csv, 'w') as f:
        w = csv.DictWriter(f, dataset_dict.keys())
        w.writeheader()
        w.writerow(dataset_dict)

    # POST dataset.csv
    dataset_csv_file = [('dataset.csv', open(dataset_csv,'rb'))]

    update_post_params = dict()
    update_post_params['sessionid'] = SESSION_ID
    update_post_params['status'] = "In Progress"

    headers = {}
    response = requests.request(
        "POST", UPDATE_DATASET_URL, headers=headers, data=update_post_params, files=dataset_csv_file
    )
    return response.content


class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])    
