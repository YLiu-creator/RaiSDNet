import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s,  CUDA_VISIBLE_DEVICES: %s\n" % (device, '5'))

from tqdm import tqdm
import metrics as M
import numpy as np
from Modules.utils.data_bjtu import test_dataset
import torch.backends.cudnn as cudnn
import cv2
import argparse
from Modules.models.RaiSDNet import Swin_RaiSDNet

# test function
def val(test_loader, model,save_path):
    global best_mae, best_epoch
    with torch.no_grad():
        mae_sum = 0
        for i in tqdm(range(test_loader.size)):
            image, gt, depth, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt = gt / (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1, 3, 1, 1).cuda()

            seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_seg = model(image, depth)

            res = seg_pre_fusion.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)


            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])

            FM.step(pred=res * 255, gt=gt * 255)
            WFM.step(pred=res * 255, gt=gt * 255)
            SM.step(pred=res * 255, gt=gt * 255)
            EM.step(pred=res * 255, gt=gt * 255)
            MAE.step(pred=res * 255, gt=gt * 255)

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
        precision = str(pre.mean().round(6))
        recall = str(rec.mean().round(6))

        eval_record = str('Dataset:' + Dataset_r + '\n||' +
                          'MAE:' + MAE_r + ';    ' +
                          'Smeasure:' + Smeasure_r + ';     ' +
                          'maxFm:' + maxFm_r + ';    ' +
                          'weiFm:' + Wmeasure_r + ';    ' +
                          'maxEm:' + maxEm_r + ';   ' +
                          'Precision:' + precision + ';   ' +
                          'Recall:' + recall + ';   ')

        print(eval_record)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--imagesize', type=int, default=587, help='training dataset size')

    parser.add_argument('--img_root', type=str, default='./ORSD_Dataset/image/',
                        help='the training texture images root')
    parser.add_argument('--dep_root', type=str, default='./ORSD_Dataset/depth/', 
                        help='the training depth images root')
    parser.add_argument('--label_root', type=str, default='./ORSD_Dataset/label/', 
                        help='the training gt images root')

    parser.add_argument('--test_img_root', type=str, default='./ORSD_Dataset/image-test/',
                        help='the test rgb images root')
    parser.add_argument('--test_dep_root', type=str, default='./ORSD_Dataset/depth-test/',
                        help='the test depth images root')
    parser.add_argument('--test_gt_root', type=str, default='./ORSD_Dataset/label-test/',
                        help='the test gt images root')

    parser.add_argument('--ckpt_path', type=str,default=r'./bestCKPT_RaiSDNet_epoch_XXX.pth',
                        help='the path to save models and logs')

    parser.add_argument('--save_path', type=str, default=r'./Results/', help='the path to save models and logs')

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

    train_loader = test_dataset(train_img_root, train_gt_root, train_dep_root, opt.imagesize)
    test_loader = test_dataset(val_img_root, val_gt_root, val_dep_root, opt.imagesize)

    model = Swin_RaiSDNet(in_chans=3, num_classes=2)
    checkpoint = torch.load(opt.ckpt_path)
    model.load_state_dict(checkpoint, strict=True)
    print('load model from ', opt.ckpt_path)
    model.cuda()
    model.eval()


    print(len(train_loader))
    print("Start test...")
    FM = M.Fmeasure_and_FNR()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    val(test_loader, model, save_path)

