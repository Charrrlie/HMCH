import os
import numpy as np
import pickle
from datetime import datetime
from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter  

from config import DefaultConfig
from models import ImgModule, TxtModule, ClassifierModule
from utils import calc_map_k, load_data, split_data

opt = DefaultConfig()

def calc_loss(img_model, txt_model, cls_model, X, Y, B, F, G, L, cls_criterion):
    """
        B (num_train, bit)
        F (num_train, bit)
        G (num_train, bit)
        L (num_train, num_class1)
    """
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)

    loss_cls = 0.0

    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        text = Y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)

        label = Variable(L[ind]) 

        if image.shape[0] == 0 or text.shape[0] == 0:
            continue

        if opt.use_gpu:
            image = image.cuda()
            text = text.cuda()
            label = label.cuda()

        with torch.no_grad():
            cur_f = img_model(image)
            predict_f = cls_model(cur_f)
            cur_g = txt_model(text)
            predict_g = cls_model(cur_g)
            loss_cls += (cls_criterion(predict_f, label) + cls_criterion(predict_g, label))*batch_size
            
    with torch.no_grad():
        quant_loss = opt.alpha * (torch.sum(torch.pow(B - F, 2) + torch.pow(B - G, 2)))
        balance_loss = opt.beta * torch.sum(torch.pow(F.sum(dim=0), 2) + torch.pow(G.sum(dim=0), 2))

    loss = loss_cls + quant_loss + balance_loss
    print(loss_cls, quant_loss, balance_loss)
    print("loss: ", loss)
    return loss


def generate_image_code(img_model, X, bit):
    batch_size = opt.batch_size
    num_data = X.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)

    if opt.use_gpu:
        B = B.cuda()

    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        image = X[ind].type(torch.float)
        if image.shape[0] == 0 :
            continue
        if opt.use_gpu:
            image = image.cuda()
        cur_f = img_model(image)
        B[ind, :] = cur_f.data
    B = torch.sign(B)

    return B


def generate_text_code(txt_model, Y, bit):
    batch_size = opt.batch_size
    num_data = Y.shape[0]
    index = np.linspace(0, num_data - 1, num_data).astype(int)
    B = torch.zeros(num_data, bit, dtype=torch.float)
    if opt.use_gpu:
        B = B.cuda()
    for i in range(num_data // batch_size + 1):
        ind = index[i * batch_size: min((i + 1) * batch_size, num_data)]
        text = Y[ind].unsqueeze(1).unsqueeze(-1).type(torch.float)
        if opt.use_gpu:
            text = text.cuda()
        cur_g = txt_model(text)
        B[ind, :] = cur_g.data
    B = torch.sign(B)

    return B


def train(**kwargs):
    opt.parse(kwargs)
    
    writer = SummaryWriter('./log')

    # load data
    images, tags, labels = load_data(opt.data_path)

    y_dim = tags.shape[1]
    X, Y, L = split_data(images, tags, labels)
    print('...loading and splitting data finish')

    # init module
    img_model = ImgModule(opt.bit)
    txt_model = TxtModule(y_dim, opt.bit)
    cls1_model = ClassifierModule(opt.bit, opt.num_class1)
    cls2_model = ClassifierModule(opt.bit, opt.num_class2) 

    if opt.use_gpu:
        img_model = img_model.cuda()
        txt_model = txt_model.cuda()
        cls1_model = cls1_model.cuda()
        cls2_model = cls2_model.cuda()

    if opt.load_model:
        print("load trained model from file..")
        img_model.load(opt.load_img_path, use_gpu=True)
        txt_model.load(opt.load_txt_path, use_gpu=True)
        cls1_model.load(opt.load_cls1_path, use_gpu=True)
        cls2_model.load(opt.load_cls2_path, use_gpu=True)

    # training algorithm
    train_L = torch.from_numpy(L['train']).float() # (num_train, num_class1)　one-hot
    train_L1 = torch.from_numpy(L['train'][:, 0:opt.num_class1]).float() # (num_train, num_class1) 
    train_L2 = torch.from_numpy(L['train'][:, opt.num_class1:opt.num_class]).float() # (num_train, num_class2) 
    train_label = torch.argmax(train_L, -1) # number label    
    train_label1 = torch.argmax(train_L1, -1) # label
    train_label2 = torch.argmax(train_L2, -1) # label

    print("train label shape", train_L.shape)
    train_x = torch.from_numpy(X['train']).float()
    train_y = torch.from_numpy(Y['train']).float()

    query_L = torch.from_numpy(L['query']).float()
    query_x = torch.from_numpy(X['query']).float()
    query_y = torch.from_numpy(Y['query']).float()

    eval_query_L = torch.from_numpy(L['eval_query']).float()
    eval_query_x = torch.from_numpy(X['eval_query']).float()
    eval_query_y = torch.from_numpy(Y['eval_query']).float()

    retrieval_L = torch.from_numpy(L['retrieval']).float()
    retrieval_x = torch.from_numpy(X['retrieval']).float()
    retrieval_y = torch.from_numpy(Y['retrieval']).float()

    num_train = train_x.shape[0]

    F_buffer = torch.randn(num_train, opt.bit) # tensor (num_train, bit)
    G_buffer = torch.randn(num_train, opt.bit) # tensor (num_train, bit)

    if opt.use_gpu:
        train_L = train_L.cuda()
        train_L1 = train_L1.cuda()
        train_L2 = train_L2.cuda()
        train_label = train_label.cuda()
        train_label1 = train_label1.cuda()
        train_label2 = train_label2.cuda()
        F_buffer = F_buffer.cuda()
        G_buffer = G_buffer.cuda()
    
    B = torch.sign(F_buffer + G_buffer) # tensor (num_train, bit)

    # optimizers 
    optimizer_img = Adam(img_model.parameters(), lr=opt.lr)
    optimizer_txt = Adam(txt_model.parameters(), lr=opt.lr)
    optimizer_cls1 = Adam(cls1_model.parameters(), lr=opt.lr) 
    optimizer_cls2 = Adam(cls2_model.parameters(), lr=opt.lr)

    cls_criterion = nn.CrossEntropyLoss()

    mapi2t, mapt2i = evaluate(img_model, txt_model, eval_query_x, eval_query_y, retrieval_x, retrieval_y, eval_query_L, retrieval_L, opt.bit)
    print("{}".format(datetime.now()))
    print('...test map: map(i->t): \033[1;32;40m%3.3f\033[0m, map(t->i): \033[1;32;40m%3.3f\033[0m' % (mapi2t, mapt2i))

    print('...training procedure starts')
    best_mapi2t = 0.0
    
    ones = torch.ones(opt.batch_size, 1)
    ones_ = torch.ones(num_train - opt.batch_size, 1)
    if opt.use_gpu:
        ones = ones.cuda()
        ones_ = ones_.cuda()

    for epoch in range(1, opt.max_epoch+1):
        print("Epoch: ", epoch)
        print("Task  1 ---")
        
        # === loss 1 task ===
        if opt.cal_loss:
            loss_1 = calc_loss(img_model, txt_model, cls1_model, train_x, train_y, B, F_buffer, G_buffer, train_label1, cls_criterion)
            writer.add_scalar('Task 1 loss', loss_1, epoch)

        # === train image net & update F & classifier1
        for i in tqdm(range(num_train // opt.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            label1 = Variable(train_label1[ind]) # (batch_size) 
        
            image = Variable(train_x[ind].type(torch.float)) # (batch_size, 224, 224, 3) no gradient
            if opt.use_gpu:
                image = image.cuda()
                label1 = label1.cuda()

            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.data  # update F

            # f 生成标签
            predict_l1 = cls1_model(cur_f)
            loss_cls1_x = cls_criterion(predict_l1, label1)
            
            # ||B-f||_2^F
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2)) / (opt.batch_size * num_train)
            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F_buffer[unupdated_ind].t().mm(ones_), 2)) / (opt.batch_size * num_train)
            loss_x = loss_cls1_x + opt.alpha * quantization_x + opt.beta * balance_x
            
            optimizer_cls1.zero_grad()
            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            optimizer_cls1.step()
        
        # === train txt net & update G & classifier1
        for i in tqdm(range(num_train // opt.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)

            label1 = Variable(train_label1[ind]) # (batch_size)
        
            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                label1 = label1.cuda()
            
            cur_g = txt_model(text) # cur_g: (batch_size, bit)
            G_buffer[ind, :] = cur_g.data   # update G

            # g 生成标签
            predict_l1 = cls1_model(cur_g)
            loss_cls1_y = cls_criterion(predict_l1, label1)
            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))  / (opt.batch_size * num_train)
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G_buffer[unupdated_ind].t().mm(ones_), 2)) / (num_train * opt.batch_size)

            loss_y = loss_cls1_y + opt.alpha*quantization_y + opt.beta*balance_y

            optimizer_cls1.zero_grad()
            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            optimizer_cls1.step()
        
        # === update B
        B = torch.sign(F_buffer + G_buffer)

        # === loss 2 task ===
        if opt.cal_loss:
            loss_2 = calc_loss(img_model, txt_model, cls2_model, train_x, train_y, B, F_buffer, G_buffer, train_label2, cls_criterion)
            writer.add_scalar('Task 2 loss', loss_2, epoch)

        # ======== train l2  
        print("Task 2 ---")
        # === train image net & update F & classifier1
        for i in tqdm(range(num_train // opt.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
            
            label2 = Variable(train_label2[ind]) # (batch_size)

            image = Variable(train_x[ind].type(torch.float)) # (batch_size, 224, 224, 3)
            if opt.use_gpu:
                image = image.cuda()
                label2 = label2.cuda()

            cur_f = img_model(image)  # cur_f: (batch_size, bit)
            F_buffer[ind, :] = cur_f.data  # update F

            # f 生成标签
            predict_l2 = cls2_model(cur_f)
            loss_cls2_x = cls_criterion(predict_l2, label2)
            
            # ||B-f||_2^F
            quantization_x = torch.sum(torch.pow(B[ind, :] - cur_f, 2))  / (opt.batch_size * num_train)

            balance_x = torch.sum(torch.pow(cur_f.t().mm(ones) + F_buffer[unupdated_ind].t().mm(ones_), 2)) / (opt.batch_size * num_train)

            loss_x = loss_cls2_x + opt.alpha*quantization_x + opt.beta*balance_x
            
            optimizer_cls2.zero_grad()
            optimizer_img.zero_grad()
            loss_x.backward()
            optimizer_img.step()
            optimizer_cls2.step()
        
        # === train txt net & update G & classifier1
        for i in tqdm(range(num_train // opt.batch_size)):
            index = np.random.permutation(num_train)
            ind = index[0: opt.batch_size]
            unupdated_ind = np.setdiff1d(range(num_train), ind)
 
            label2 = Variable(train_label2[ind]) # (batch_size)

            text = train_y[ind, :].unsqueeze(1).unsqueeze(-1).type(torch.float)
            text = Variable(text)
            if opt.use_gpu:
                text = text.cuda()
                label2 = label2.cuda()
            
            cur_g = txt_model(text) # cur_g: (batch_size, bit)
            G_buffer[ind, :] = cur_g.data   # update G

            # g 生成标签
            predict_l2 = cls2_model(cur_g)
            loss_cls2_y = cls_criterion(predict_l2, label2)

            quantization_y = torch.sum(torch.pow(B[ind, :] - cur_g, 2))  / (opt.batch_size * num_train)
            balance_y = torch.sum(torch.pow(cur_g.t().mm(ones) + G_buffer[unupdated_ind].t().mm(ones_), 2)) / (num_train * opt.batch_size)

            loss_y = loss_cls2_y + opt.alpha*quantization_y + opt.beta*balance_y
        
            optimizer_cls2.zero_grad()
            optimizer_txt.zero_grad()
            loss_y.backward()
            optimizer_txt.step()
            optimizer_cls2.step()
        
        # === update B
        B = torch.sign(F_buffer + G_buffer)

        if epoch % 1 == 0:
            mapi2t, mapt2i = evaluate(img_model, txt_model, eval_query_x, eval_query_y, retrieval_x, retrieval_y, eval_query_L, retrieval_L, opt.bit)
            
            writer.add_scalar('mAP(i2t)', mapi2t, epoch)
            writer.add_scalar('mAP(t2i)', mapt2i, epoch)

            # save best model
            if mapi2t > best_mapi2t:
                print("best mapi2t, save model...")
                best_mapi2t = mapi2t
                txt_model.save(opt.load_txt_path)
                img_model.save(opt.load_img_path)
                cls1_model.save(opt.load_cls1_path)
                cls2_model.save(opt.load_cls2_path)

            print("{}".format(datetime.now()))
            print("Epoch: ", epoch)
            print('...eval map: map(i->t): \033[1;32;40m%3.3f\033[0m, map(t->i): \033[1;32;40m%3.3f\033[0m'%(mapi2t, mapt2i))
            
            mapi2t, mapt2i = evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, opt.bit)
            print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))

    print('...training procedure finish')
    print("Retrieval begin:")
    print("{}".format(datetime.now()))
    mapi2t, mapt2i = evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, opt.bit)
    print('...test map: map(i->t): %3.3f, map(t->i): %3.3f' % (mapi2t, mapt2i))
    print("{}".format(datetime.now()))
    print("Retrieval end:")


def evaluate(img_model, txt_model, query_x, query_y, retrieval_x, retrieval_y, query_L, retrieval_L, bit):
    qBX = generate_image_code(img_model, query_x, opt.bit)
    qBY = generate_text_code(txt_model, query_y, opt.bit)
    rBX = generate_image_code(img_model, retrieval_x, opt.bit)
    rBY = generate_text_code(txt_model, retrieval_y, opt.bit)

    mapi2t = calc_map_k(qBX, rBY, query_L, retrieval_L)
    mapt2i = calc_map_k(qBY, rBX, query_L, retrieval_L)

    return mapi2t, mapt2i


if __name__ == '__main__':
    train()
