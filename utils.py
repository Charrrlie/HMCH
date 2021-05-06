import scipy.io 
import numpy as np
import torch

from config import DefaultConfig

opt = DefaultConfig()


def load_data(path):
    image_path = path + "image.mat"
    label_path = path + "label.mat"
    tag_path = path + "tag.mat"

    images = scipy.io.loadmat(image_path)['Image']   # [19862, 224, 224, 3]
    tags = scipy.io.loadmat(tag_path)['Tag']     # [19862, 2685]
    labels = scipy.io.loadmat(label_path)["Label"]    # [19862, 35]

    return images, tags, labels


def split_data(images, tags, labels):
    # shuffle data
    np.random.seed(opt.seed)
    shuffle_idx = np.random.permutation(np.arange(len(images)))
    images = images[shuffle_idx]
    tags = tags[shuffle_idx]
    labels = labels[shuffle_idx]
    
    X = {}
    X['query'] = images[0: opt.query_size]
    X['eval_query'] = images[opt.query_size: 2*opt.query_size]
    X['train'] = images[opt.query_size: opt.training_size + opt.query_size]
    X['retrieval'] = images[opt.query_size: opt.query_size + opt.database_size]

    Y = {}
    Y['query'] = tags[0: opt.query_size]
    Y['eval_query'] = tags[opt.query_size: 2*opt.query_size]
    Y['train'] = tags[opt.query_size: opt.training_size + opt.query_size]
    Y['retrieval'] = tags[opt.query_size: opt.query_size + opt.database_size]
    
    L = {}
    L['query'] = labels[0: opt.query_size]
    L['eval_query'] = labels[opt.query_size: 2*opt.query_size]
    L['train'] = labels[opt.query_size: opt.training_size + opt.query_size]
    L['retrieval'] = labels[opt.query_size: opt.query_size + opt.database_size]

    return X, Y, L


def calc_hammingDist(B1, B2):
    """calculate Hamming distance
    """
    bit = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (bit - B1.mm(B2.transpose(0, 1)))

    return distH


def calc_map_k(qB, rB, query_L, retrieval_L, k=None): 
    """
    MAP: mean Average Precision
        mAP = 1/|Q_R| sum_q in Q_R AP(q)

        qB: {-1,+1}^{M x bit} tensor
        rB: {-1,+1}^{N x bit} tensor
        query_L: {0,1}^{M x num_class2} tensor
        retrieval_L: {0,1}^{N x num_class2} tensor
    """
    num_query = query_L.shape[0] # 3000
    mAP = 0
    if k is None:
        k = retrieval_L.shape[0] # 16862
    
    # only label class_2 count
    query_L = query_L[ : , opt.num_class1 : opt.num_class]  
    retrieval_L = retrieval_L[ : , opt.num_class1 : opt.num_class]

    for it in range(num_query):
        q_L = query_L[it]   # （num_class2)
        if len(q_L.shape) < 2:
            q_L = q_L.unsqueeze(0) # (1, num_class2)

        gnd = (q_L.mm(retrieval_L.transpose(0, 1)) > 0).squeeze().type(torch.float32) # N
        tsum = torch.sum(gnd) # ground truth num
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[it, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_() # (N)
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float32)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float32) + 1.0
        if tindex.is_cuda:
            count = count.cuda()
        mAP = mAP + torch.mean(count / tindex)
    mAP = mAP / num_query

    return mAP

