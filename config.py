import warnings

class DefaultConfig(object):
    load_model = False
    load_img_path = './checkpoints/image_model.pth'  # load model path
    load_txt_path = './checkpoints/text_model.pth'
    load_cls1_path = './checkpoints/cls_model1.pth'
    load_cls2_path = './checkpoints/cls_model2.pth'

    vgg_path = './data/imagenet-vgg-f.mat'

    # === data parameters
    is_FashionVC = True
    if is_FashionVC:
        data_path = './data/FashionVC/'
        training_size = 16862
        query_size = 3000
        database_size = 16862

        num_class1 = 8
        num_class2 = 27

        num_class = num_class1 + num_class2
    else:
        data_path = './data/Ssense/'
        training_size = 13696
        query_size = 2000
        database_size = 13696

        num_class1 = 4
        num_class2 = 28

        num_class = num_class1 + num_class2
    
    bit = 32

    # === hyper-parameters
    max_epoch = 30
    batch_size = 128

    alpha = 50 # quantization loss
    beta = 0.1 # balance loss
    lr = 1e-4  # initial learning rate
    seed = 1234 # random seed

    use_gpu = True
    valid = True
    cal_loss = True

    result_dir = 'result'

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('User config:')
        for k, v in self.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

