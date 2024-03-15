
import os
import pandas as pd
import json

def collect_kaggle(path):
    npy_datas_x = sorted(os.listdir(path + "image"))
    npy_datas_y = sorted(os.listdir(path + "depth"))

    if '.DS_Store' in npy_datas_x:
        npy_datas_x.remove('.DS_Store')

    if '.DS_Store' in npy_datas_y:
        npy_datas_y.remove('.DS_Store')

    data_paths = []
    label_paths = []
    for i in range(len(npy_datas_x)):
        data_path = path + "image/" + npy_datas_x[i]
        label_path = path + "depth/" + npy_datas_y[i]

        assert(npy_datas_x[i] == npy_datas_y[i]), "label and data not same!"

        data_paths.append(data_path.encode("ascii"))
        label_paths.append(label_path.encode("ascii"))

    return [
        {
            "x_file":data,
            "y_file":label
        }
        for data, label in zip(data_paths, label_paths)
    ]


def collect_cityspace(path):
    
    meta_data = pd.read_csv(path)

    image_paths = sorted(meta_data["image"].to_list())
    depth_paths = sorted(meta_data["depth"].to_list())
    camera_paths = sorted(meta_data["camera"].to_list())


    data_paths = []
    label_paths = []
    depth_parms = []

    for i in range(len(camera_paths)):

        with open(camera_paths[i], "r") as read_file:
            camera_params = json.load(read_file)

        depth_parm = camera_params['extrinsic']['baseline'] * camera_params['intrinsic']['fx']
        data_paths.append(image_paths[i].encode("ascii"))
        label_paths.append(depth_paths[i].encode("ascii"))
        depth_parms.append(depth_parm)

    return [
        {
            "x_file":data,
            "y_file":label,
            "camera":camera
        }
        for data, label, camera in zip(data_paths, label_paths, depth_parms)
    ]