import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim
from net.model import SS_UIE_model
from utils.utils import *
from utils.LAB import *
from utils.LCH import *
from utils.FDL import *
import cv2
import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv
import random
import torch.utils.data as dataf
import torch.nn.functional as F


dtype = 'float32'
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
#torch.cuda.set_device(1)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_default_tensor_type(torch.FloatTensor)

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'  # 使用多张 GPU
device_ids = [0,1,2]
torch.set_default_tensor_type(torch.FloatTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def sample_images(batches_done):
    """Saves a generated sample from the validation set"""

    SS_UIE.eval()
    i=random.randrange(1,500)
    real_A = Variable(x_test[i,:,:,:]).cuda()
    real_B = Variable(Y_test[i,:,:,:]).cuda()
    real_A=real_A.unsqueeze(0)
    real_B=real_B.unsqueeze(0)
    fake_B = SS_UIE(real_A)
    #print(fake_B.shape)
    imgx=fake_B.data
    imgy=real_B.data
    x=imgx[:,:,:,:]
    y=imgy[:,:,:,:]
    img_sample = torch.cat((x,y), -2)
    save_image(img_sample, "images/%s/%s.png" % ('results', batches_done), nrow=5, normalize=True)#要改




training_x=[]
path='./data/Train/input/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    training_x.append(imgx)   

X_train = []
for features in training_x:
    X_train.append(features)

X_train = np.array(X_train)
X_train=X_train.astype(dtype)
X_train= torch.from_numpy(X_train)
X_train=X_train.permute(0,3,1,2)
#X_train=X_train.unsqueeze(1)
X_train=X_train/255.0
print("input shape:",X_train.shape)


training_y=[]
path='./data/Train/GT/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    training_y.append(imgx)


y_train = []
for features in training_y:
    y_train.append(features)

y_train = np.array(y_train)
y_train=y_train.astype(dtype)
y_train= torch.from_numpy(y_train)
y_train=y_train.permute(0,3,1,2)
y_train=y_train/255.0
print("output shape:",y_train.shape)


test_x=[]
path='./data/Test/input/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    test_x.append(imgx)


x_test = []
for features in test_x:
    x_test.append(features)

x_test = np.array(x_test)
x_test=x_test.astype(dtype)
x_test= torch.from_numpy(x_test)
x_test=x_test.permute(0,3,1,2)
x_test=x_test/255.0
print("test input shape:",x_test.shape)

test_Y=[]
path='./data/Test/GT/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    test_Y.append(imgx)


Y_test = []
for features in test_Y:
    Y_test.append(features)

Y_test = np.array(Y_test)
Y_test=Y_test.astype(dtype)
Y_test= torch.from_numpy(Y_test)
Y_test=Y_test.permute(0,3,1,2)
Y_test=Y_test/255.0
print("test output shape:",Y_test.shape)





dataset = dataf.TensorDataset(X_train,y_train)
loader = dataf.DataLoader(dataset, batch_size=9, shuffle=True,num_workers=4)
SS_UIE = SS_UIE_model(in_channels=3, channels=16, num_resblock=4, num_memblock=4).cuda()
SS_UIE = torch.nn.DataParallel(SS_UIE, device_ids=device_ids)



#net.apply(weights_init_kaiming)
MSE= nn.L1Loss(size_average=False).cuda()
SSIM = pytorch_ssim.SSIM().cuda()
L_lab=lab_Loss().cuda()
L_lch=lch_Loss().cuda()
FDL_loss = FDL(loss_weight=1.0,alpha=2.0,patch_factor=4,ave_spectrum=True,log_matrix=True,batch_matrix=True).cuda()



LR=0.0001

optimizer = torch.optim.Adam(SS_UIE.parameters(), lr=LR, betas=(0.5, 0.999))
scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=400,gamma=0.8)



use_pretrain=False
if use_pretrain:
    # Load pretrained models
    start_epoch=967
    SS_UIE.load_state_dict(torch.load("saved_models/SS_UIE_%d.pth" % (start_epoch)))
    print('successfully loading epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 0
    print('No pretrain model found, training will start from scratch！')


# ----------
#  Training
# ----------
f1 = open('psnr.csv','w',encoding='utf-8')#要改
csv_writer1 = csv.writer(f1)
f2 = open('SSIM.csv','w',encoding='utf-8')#要改
csv_writer2 = csv.writer(f2)

checkpoint_interval=5
epochs=start_epoch
n_epochs=2000
sample_interval=1000

# ingnored when opt.mode=='S'
psnr_max=0
psnr_list = [] 
prev_time = time.time()

for epoch in range(epochs,n_epochs):
    psnr_list = []
    for i, batch in enumerate(loader):

        # Model inputs
        Input = Variable(batch[0]).cuda().contiguous() 
        GT = Variable(batch[1]).cuda().contiguous()



        # ------------------
        #  Train 
        # ------------------

        optimizer.zero_grad()

        # loss
        output = SS_UIE(Input)
        loss_RGB= MSE(output, GT)/(GT.size()[2]**2)
        loss_lab = (L_lab(output, GT)+L_lab(output, GT)+L_lab(output, GT)+L_lab(output, GT))/4.0
        loss_lch = (L_lch(output, GT)+L_lch(output, GT)+L_lch(output, GT)+L_lch(output, GT))/4.0    
        loss_ssim=1-SSIM(output,GT)
        ssim_value = -(loss_ssim.item()-1)
        fdl_loss = FDL_loss(output, GT)





        loss_final=loss_ssim*10+loss_RGB*10+loss_lch+loss_lab*0.0001+fdl_loss*10000



        loss_final.backward(retain_graph=True)

        optimizer.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(loader) + i
        batches_left = n_epochs * len(loader) - batches_done
        out_train= torch.clamp(output, 0., 1.) 
        psnr_train = batch_PSNR(out_train,GT, 1.)
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if batches_done%100==0:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %2f] [SSIM: %2f][loss: %2f][loss_lch: %2f][loss_lab: %2f][fdl_loss: %2f] ETA: %2s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(loader),
                    psnr_train,
                    ssim_value,
                    loss_final.item(),
                    loss_lch.item(),
                    loss_lab.item()*0.0001,
                    fdl_loss.item()*5000, 
                    time_left,
                )
            )


        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done)
            csv_writer1.writerow([str(psnr_train)])
            csv_writer2.writerow([str(ssim_value)])
        psnr_list.append(psnr_train)

    PSNR_epoch=np.array(psnr_list)
    if PSNR_epoch.mean()>psnr_max:
        torch.save(SS_UIE.state_dict(), "saved_models/SS_UIE_%d.pth" % (epoch))
        psnr_max=PSNR_epoch.mean()
        print("")
        print('A checkpoint Saved PSNR= %f'%(psnr_max))


    scheduler.step()
#    if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
#        # Save model checkpoints
#        torch.save(memnet.state_dict(), "saved_models/uie-memnet_%d.pth" % (epoch))