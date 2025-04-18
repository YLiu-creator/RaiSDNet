import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, '5'))

from tqdm import tqdm
import metrics as M
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from Modules.utils.data_bjtu import get_loader, test_dataset
from Modules.utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from Modules.pytorch_iou import IOU
import argparse
from Modules.models.RaiSDNet import Swin_RaiSDNet


def train(train_loader, model, optimizer, epoch,save_path):
    global step
    model.train()
    loss_all=0
    epoch_step=0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images   = images.cuda()                    #3,256,256
            gts      = gts.cuda()                       #1,256,256
            depths   = depths.repeat(1,3,1,1).cuda()    #3,256,256

            B = opt.batchsize
            cls_lbl,_ = torch.max(gts.view(B, -1), dim=1)

            gts_256 =F.interpolate(gts, size=(256, 256), mode='nearest')
            gts_128 =F.interpolate(gts, size=(128, 128), mode='nearest')

            seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_seg = model(images, depths)

            loss_img = CE(seg_pre_img, gts_128) + IOU(seg_pre_img, gts_128)
            loss_dep = CE(seg_pre_dep, gts_128) + IOU(seg_pre_dep, gts_128)
            loss_seg = CE(seg_pre_fusion, gts) + IOU(seg_pre_fusion, gts)

            loss1 = CE(hidden_seg[0], gts)+IOU(hidden_seg[0], gts)
            loss2 = CE(hidden_seg[1], gts_256)+IOU(hidden_seg[1], gts_256)
            loss3 = CE(hidden_seg[2], gts_128)+IOU(hidden_seg[2], gts_128)

            loss = loss_img+loss_dep+loss_seg+loss1+loss2+loss3
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step+=1
            epoch_step+=1
            loss_all+=loss.data

            if i % 50 == 0 or i == total_step or i==1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], L_img: {:.4f}  L_dep: {:.4f} L_fuse: {:.4f} '
                      'Layer1: {:.4f} Layer2: {:.4f}  Layer3: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           loss_img.data,loss_dep.data,loss_seg.data, loss1.data,loss2.data,loss3.data))
                logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], L_img: {:.4f}  L_dep: {:.4f}  L_fuse: {:.4f} '
                             'Layer1: {:.4f}  Layer2: {:.4f}  Layer3: {:.4f}'.
                    format( epoch, opt.epoch, i, total_step,
                            loss_img.data,loss_dep.data,loss_seg.data, loss1.data,loss2.data,loss3.data))
                
        loss_all/=epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format( epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        
        if (epoch) % 20 == 0:
            torch.save(model.state_dict(), save_path+'RaiSDNet_epoch_{}.pth'.format(epoch))
            
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path+'RaiSDNet_epoch_{}.pth'.format(epoch+1))
        print('save checkpoints successfully!')
        raise
        
        
        
#test function
def val(test_loader,model,epoch,save_path):
    global best_mae,best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in tqdm(range(test_loader.size)):
            image, gt, depth, name,img_for_post = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.repeat(1,3,1,1).cuda()

            seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_seg = model(image,depth)

            res = seg_pre_fusion.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

            FM.step(pred=res*255, gt=gt*255)
            WFM.step(pred=res*255, gt=gt*255)
            SM.step(pred=res*255, gt=gt*255)
            EM.step(pred=res*255, gt=gt*255)
            MAE.step(pred=res*255, gt=gt*255)

        print('Test Done!')
        fm_res = FM.get_results()
        pre = fm_res[0]['pr']['p']
        rec = fm_res[0]['pr']['r']
        fm = fm_res[0]['fm']
        wfm = WFM.get_results()['wfm']
        sm = SM.get_results()['sm']
        em = EM.get_results()['em']
        mae = MAE.get_results()['mae']

        Dataset_r = str(opt.save_path)
        Smeasure_r = str(sm.round(6))
        Wmeasure_r = str(wfm.round(6))
        MAE_r = str(mae.round(6))
        maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(6))
        maxFm_r = str(fm['curve'].max().round(6))
        precision =  str(pre.mean().round(6))
        recall =  str(rec.mean().round(6))

        eval_record = str('Dataset:' + Dataset_r + '\n||' +
                          'MAE:' + MAE_r + ';    ' +
                          'Smeasure:' + Smeasure_r + ';     ' +
                          'maxFm:' + maxFm_r + ';    ' +
                          'weiFm:' + Wmeasure_r + ';    ' +
                          'maxEm:' + maxEm_r + ';   ' +
                          'Precision:' + precision + ';   ' +
                          'Recall:' + recall + ';   ')

        print(eval_record)

        mae = mae_sum/test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch,mae,best_mae,best_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path+'bestCKPT_RaiSDNet_epoch_{}.pth'.format(str(best_mae)[2:7]))
                print('best epoch:{}'.format(epoch))
                
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch,mae,best_epoch,best_mae))
 
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--imagesize', type=int, default=587, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--lw', type=float, default=0.001, help='weight')
    parser.add_argument('--decay_rate', type=float, default=0.85, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=20, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

    parser.add_argument('--img_root', type=str, default='./ORSD_Dataset/image/', help='the training texture images root')
    parser.add_argument('--dep_root', type=str, default='./ORSD_Dataset/depth/', help='the training depth images root')
    parser.add_argument('--label_root', type=str, default='./ORSD_Dataset/label/',help='the training gt images root')

    parser.add_argument('--test_img_root', type=str, default='./ORSD_Dataset/image-test/',help='the test rgb images root')
    parser.add_argument('--test_dep_root', type=str, default='./ORSD_Dataset/depth-test/',help='the test depth images root')
    parser.add_argument('--test_gt_root', type=str, default='./ORSD_Dataset/label-test/',help='the test gt images root')

    parser.add_argument('--save_path', type=str, default=r'./Results/',help='the path to save models and logs')

    opt = parser.parse_args()

    cudnn.benchmark = True

    # set the path
    train_img_root = opt.img_root
    train_gt_root = opt.label_root
    train_dep_root = opt.dep_root

    val_img_root = opt.test_img_root
    val_gt_root = opt.test_gt_root
    val_dep_root = opt.test_dep_root
    save_path = opt.save_path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(train_img_root, train_gt_root, train_dep_root, batchsize=opt.batchsize, imagesize=opt.imagesize)
    test_loader = test_dataset(val_img_root, val_gt_root, val_dep_root, opt.imagesize)
    total_step = len(train_loader)

    model = Swin_RaiSDNet(in_chans=3, num_classes=2)
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)

    # set loss function
    IOU = IOU(size_average=True)
    CE = torch.nn.BCEWithLogitsLoss()

    logging.basicConfig(filename=save_path + 'log.log', format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("BBSNet_unif-Train")
    logging.info("Config")
    logging.info('epoch:{};lr:{};batchsize:{};imagesize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
            opt.epoch, opt.lr, opt.batchsize, opt.imagesize, opt.clip, opt.decay_rate, opt.load, save_path,
            opt.decay_epoch))


    step = 0
    writer = SummaryWriter(save_path + 'summary')
    best_mae = 1
    best_epoch = 0

    print(len(train_loader))
    print("Start train...")
    FM = M.Fmeasure_and_FNR()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    for epoch in range(0, (opt.epoch+1)):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        print(cur_lr)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        train(train_loader, model, optimizer, epoch,save_path)
        val(test_loader,model,epoch,save_path)
