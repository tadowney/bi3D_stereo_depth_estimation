from glob import glob

# ------------------------------------------------------------------------------
#                               Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':

    all_dir = glob("../../../sceneflow/frames_finalpass/train/*/")

    f = open('./dataloaders/lists/sceneflow_train_td.list', 'w+')
    for d in all_dir:
        all_files = glob(d + 'left/*.png')
        for file in all_files:
            path = file.split('\\')
            img = path[-1]
            imgPath = 'train/' + path[1] + '/' + path[2] + '/' + img
            f.write(imgPath + '\n')
    f.close()

    all_dir = glob("../../../sceneflow/frames_finalpass/test/*/")

    f = open('./dataloaders/lists/sceneflow_test_td.list', 'w+')
    for d in all_dir:
        all_files = glob(d + 'left/*.png')
        for file in all_files:
            path = file.split('\\')
            img = path[-1]
            imgPath = 'test/' + path[1] + '/' + path[2] + '/' + img
            f.write(imgPath + '\n')
    f.close()

          