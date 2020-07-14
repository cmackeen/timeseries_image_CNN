from torch.utils.data import TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import warm
import warm.functional as W
from src.utils.resnet import *

def tensor_load(tf,lf,tfo,lfo):
    batch_s=20
    #train/val data
    dataset=TensorDataset(tf, lf.flatten())

    #out-of-window data
    dso=TensorDataset(tfo, lfo.flatten())

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    tr_set = torch.utils.data.DataLoader(train_dataset, batch_size=batch_s, shuffle=True)


    tt_set = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)
    to_set = torch.utils.data.DataLoader(dso, batch_size=5, shuffle=False)
    return tr_set,tt_set,to_set

def train(tr_set,tf_length,EPOCH = 30,LR = 0.055,batch_s=20,w_tot=60,seql=60):
    size=[batch_s, 3, w_tot, seql]

    resnet18 = ResNet()

    
    epochs=EPOCH
    BATCH_SIZE = 20
    
    DOWNLOAD_MNIST = False
    N_TEST_IMG = 5

    device = torch.device("cuda")
    model = ResNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(tr_set), epochs=EPOCH)
    criterion = nn.CrossEntropyLoss()

    lossarr=[]
    for epoch in range(EPOCH):
            loss = 0
            for fs,ls in tr_set:
                    optimizer.zero_grad()
                    outputs = model(fs)
                    #print(outputs)
                    ls=ls.long()
                    #print(ls)
                    train_loss = criterion(outputs, ls)
                    train_loss.backward()
                    optimizer.step()
                    scheduler.step()
                    loss += train_loss.item()

            loss = loss / tf_length
            lossarr.append([loss,scheduler.get_lr()])
            print(scheduler.get_lr())
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))


    return model


def model_eval(trained_model,data_set):
    correct= 0
    total = 0
    tp=0
    fp=0
    tn=0
    fn=0
    
    with torch.no_grad():
            for data in data_set:
                    X, y = data
                    output = trained_model(X)
                    #print(y)
                    #print(output)
                    for idx, i in enumerate(output):
                            #print(torch.argmax(i), y[idx])
                            if int(y[idx])==1:
                                if torch.argmax(i) == y[idx]:
                                        tp += 1
                                if torch.argmax(i) != y[idx]:
                                        fn += 1

                            if int(y[idx])==0:
                                if torch.argmax(i) == y[idx]:
                                        tn += 1
                                if torch.argmax(i) != y[idx]:
                                        fp += 1
                            total += 1

    print("tp,  fp,  tn,  fn")
    print([tp,fp,tn,fn])
    f1=(2*tp/(2*tp+fp+fn))
    print("f1 = " +str(f1))
    print("acc = " + str((tp+tn)/(tp+tn+fp+fn)))
    
def model_save(name):
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, str(name))
