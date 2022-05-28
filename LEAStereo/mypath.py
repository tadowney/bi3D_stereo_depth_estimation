class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'sceneflow':
            return '../sceneflow/'
        elif dataset == 'kitti15':
            return '../kitti2015/training/'
        elif dataset == 'kitti12':
            return './dataset/kitti2012/training/'
        elif dataset == 'middlebury':
            return './dataset/MiddEval3/trainingH/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
