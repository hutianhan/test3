import os
import tqdm
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from data import *
from unet import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
train_data_path = r'F:\vessel seg\data\train'
val_data_path = r'F:\vessel seg\data\test'
save_path = 'train_image'


def Dice(pred, true):
    intersection = pred * true
    temp = pred + true
    smooth = 1e-8
    dice_score = 2 * intersection.sum() / (temp.sum() + smooth)
    return dice_score

if __name__ == '__main__':

    train_data_loader = DataLoader(MyDataset(train_data_path), batch_size=4, shuffle=True)
    val_data_loader= DataLoader(MyDataset(val_data_path), batch_size=4, shuffle=True)
    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')
    lr = 0.001
    learning_rate_decay_start = 20
    learning_rate_decay_every = 10
    learning_rate_decay_rate = 0.9
    opt = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=1e-4)
    loss_fun = nn.BCELoss()
    max_dice = 0
    max_epoch = 0
    epoch = 1
    while epoch < 300:
        train_num_correct = 0
        train_num_pixels = 0
        val_num_correct = 0
        val_num_pixels = 0
        train_Dice = 0
        train_Iou = 0
        val_Dice = 0
        val_Iou = 0
        train_cnt = 0
        val_cnt = 0
        net.train()
        for i, (image, segment_image) in enumerate(tqdm.tqdm(train_data_loader)):
            image, segment_image = image.to(device), segment_image.to(device)
            out_image = net(image)
            train_loss = loss_fun(out_image,segment_image)
            opt.zero_grad()
            train_loss.backward()
            opt.step()
            out_image=(out_image>0.5).float()
            train_num_correct +=(out_image== segment_image).sum()
            train_num_pixels += torch.numel(out_image)
            out_image=out_image.reshape(1,-1)
            segment_image=segment_image.reshape(1,-1)
            train_Dice+=Dice(out_image, segment_image)
            train_cnt +=1

        print(f'{epoch}-{i}-train_Dice===>>{train_Dice/train_cnt}')

        net.eval()
        with torch.no_grad():
            for i, (image, segment_image) in enumerate(tqdm.tqdm(val_data_loader)):
                image, segment_image = image.to(device), segment_image.to(device)
                out_image = net(image)
                val_loss = loss_fun(out_image, segment_image)
                out_image = (out_image > 0.5).float()
                val_num_correct += (out_image == segment_image).sum()
                val_num_pixels += torch.numel(out_image)
                val_Dice += Dice(out_image, segment_image)
                val_cnt += 1

            print(f'{epoch}-{i}-val_Dice===>>{val_Dice / val_cnt}')
            if val_Dice / val_cnt > max_dice:
                max_dice = val_Dice / val_cnt
                max_epoch = epoch
                print('--------------------max_dice=', max_dice)
                print('--------------------max_epoch=', max_epoch)
                torch.save(net.state_dict(), weight_path)
                print('save successfully!')
        epoch += 1






