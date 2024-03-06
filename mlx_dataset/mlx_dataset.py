import os

def collect_data(path):
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