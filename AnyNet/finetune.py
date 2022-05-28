import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import KITTILoader as DA
import utils.logger as logger
import torch.backends.cudnn as cudnn
import cv2 as cv
import numpy as np
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import random

import models.anynet

def disp2rgb(disp):
    H = disp.shape[0]
    W = disp.shape[1]

    I = disp.flatten()

    map = np.array(
        [
            [0, 0, 0, 114],
            [0, 0, 1, 185],
            [1, 0, 0, 114],
            [1, 0, 1, 174],
            [0, 1, 0, 114],
            [0, 1, 1, 185],
            [1, 1, 0, 114],
            [1, 1, 1, 0],
        ]
    )
    bins = map[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]

    ind = np.minimum(
        np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None], I.shape[0], axis=1), axis=0), 6
    )
    bins = np.reciprocal(bins)
    cbins = np.append(np.array([[0]]), cbins[:, None])

    I = np.multiply(I - cbins[ind], bins[ind])
    I = np.minimum(
        np.maximum(
            np.multiply(map[ind, 0:3], np.repeat(1 - I[:, None], 3, axis=1))
            + np.multiply(map[ind + 1, 0:3], np.repeat(I[:, None], 3, axis=1)),
            0,
        ),
        1,
    )

    I = np.reshape(I, [H, W, 3]).astype(np.float32)

    return I


parser = argparse.ArgumentParser(description='Anynet fintune on KITTI')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datatype', default='2015',
                    help='datapath')
parser.add_argument('--datapath', default=None, help='datapath')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=6,
                    help='batch size for training (default: 6)')
parser.add_argument('--test_bsize', type=int, default=8,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/finetune_anynet',
                    help='the path of saving checkpoints and log')
parser.add_argument('--resume', type=str, default=None,
                    help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=10, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--start_epoch_for_spn', type=int, default=121)
parser.add_argument('--pretrained', type=str, default='results/pretrained_anynet/checkpoint.tar',
                    help='pretrained model path')
parser.add_argument('--split_file', type=str, default=None)
parser.add_argument('--evaluate', action='store_true')


args = parser.parse_args()

if args.datatype == '2015':
    from dataloader import KITTIloader2015 as ls
elif args.datatype == '2012':
    from dataloader import KITTIloader2012 as ls
elif args.datatype == 'other':
    from dataloader import diy_dataset as ls

f_loss = open("results/finetune_anynet/loss.txt", 'w')
f_val = open("results/finetune_anynet/val.txt", 'w')

def main():
    global args
    log = logger.setup_logger(args.save_path + '/training.log')

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = ls.dataloader(
        args.datapath,log, args.split_file)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    for key, value in sorted(vars(args).items()):
        log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            log.info("=> loaded pretrained model '{}'"
                     .format(args.pretrained))
        else:
            log.info("=> no pretrained model found at '{}'".format(args.pretrained))
            log.info("=> Will start from scratch.")
    args.start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            log.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            log.info("=> loaded checkpoint '{}' (epoch {})"
                     .format(args.resume, checkpoint['epoch']))
        else:
            log.info("=> no checkpoint found at '{}'".format(args.resume))
            log.info("=> Will start from scratch.")
    else:
        log.info('Not Resume')
    cudnn.benchmark = True
    start_full_time = time.time()
    if args.evaluate:
        #test(TestImgLoader, model, log)
        test_single(model, log)
        return

    for epoch in range(args.start_epoch, args.epochs):
        log.info('This is {}-th epoch'.format(epoch))
        adjust_learning_rate(optimizer, epoch)

        train(TrainImgLoader, model, optimizer, log, epoch)

        savefilename = args.save_path + '/checkpoint.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

        if epoch % 1 == 0:
            test(TestImgLoader, model, log)

    #test(TestImgLoader, model, log)
    log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))
    f_loss.close()
    f_val.close()


def train(dataloader, model, optimizer, log, epoch=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L > 0
        mask.detach_()
        outputs = model(imgL, imgR)

        if args.with_spn:
            if epoch >= args.start_epoch_for_spn:
                num_out = len(outputs)
            else:
                num_out = len(outputs) - 1
        else:
            num_out = len(outputs)

        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(num_out)]
        sum(loss).backward()
        optimizer.step()

        for idx in range(num_out):
            losses[idx].update(loss[idx].item())

        # if batch_idx % args.print_freq == 0:
        #     info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(num_out)]
        #     info_str = '\t'.join(info_str)

            # log.info('Epoch{} [{}/{}] {}'.format(
            #     epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    log.info('Average train loss = ' + info_str)

    str = "".join(["{:.2f} ".format(losses[x].avg) for x in range(stages)])
    str += "\n"
    f_loss.write(str)


def test(dataloader, model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                output = torch.squeeze(outputs[x], 1)
                D1s[x].update(error_estimating(output, disp_L).item())

        # info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])
        #
        # log.info('[{}/{}] {}'.format(
        #     batch_idx, length_loader, info_str))


    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)

    str = "".join(["{:.2f} ".format(D1s[x].avg) for x in range(stages)])
    str += "\n"
    f_val.write(str)
def test_single(model, log):

    stages = 3 + args.with_spn
    D1s = [AverageMeter() for _ in range(stages)]

    lPath = "test/left.png"
    rPath = "test/right.png"
    dPath = "test/disp.png"

    # Inputs
    img_left = Image.open(lPath).convert("RGB")
    img_right = Image.open(rPath).convert("RGB")
    img_disp = Image.open(dPath)

    img_left = transforms.functional.to_tensor(img_left)
    img_right = transforms.functional.to_tensor(img_right)
    img_disp = transforms.functional.to_tensor(img_disp)

    img_left = transforms.functional.normalize(img_left, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_right = transforms.functional.normalize(img_right, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    #img_disp = transforms.functional.normalize(img_disp, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    img_left = img_left.type(torch.cuda.FloatTensor)[None, :, :, :]
    img_right = img_right.type(torch.cuda.FloatTensor)[None, :, :, :]
    img_disp = img_disp.type(torch.cuda.FloatTensor)[None, :, :, :]

    # Pad Inputs
    tw = 1248
    th = 384
    assert tw % 96 == 0, "image dimensions should be multiple of 96"
    assert th % 96 == 0, "image dimensions should be multiple of 96"
    h = img_left.shape[2]
    w = img_left.shape[3]
    x1 = random.randint(0, max(0, w - tw))
    y1 = random.randint(0, max(0, h - th))
    pad_w = tw - w if tw - w > 0 else 0
    pad_h = th - h if th - h > 0 else 0
    pad_opr = torch.nn.ZeroPad2d((pad_w, 0, pad_h, 0))

    img_left = img_left[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
    img_right = img_right[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]
    img_disp = img_disp[:, :, y1 : y1 + min(th, h), x1 : x1 + min(tw, w)]

    img_left_pad = pad_opr(img_left)
    img_right_pad = pad_opr(img_right)
    img_disp_pad = pad_opr(img_disp)

    imgL = img_left_pad
    imgR = img_right_pad
    disp_L = img_disp_pad

    imgL = imgL.float().cuda()
    imgR = imgR.float().cuda()
    disp_L = disp_L.float().cuda()

    model.eval()
    i = 0

    with torch.no_grad():
        start_time = time.time()
        outputs = model(imgL, imgR)
        print('inference time = %.5f' %(time.time() - start_time))
        for x in range(stages):
            output = torch.squeeze(outputs[x], 1)
            D1s[x].update(error_estimating(output, disp_L).item())

    info_str = '\t'.join(['Stage {} = {:.4f}({:.4f})'.format(x, D1s[x].val, D1s[x].avg) for x in range(stages)])

    log.info('{}'.format(info_str))


    # Draw Disp Map
    _, _, H, W = outputs[0].shape
    all_results_color = torch.zeros((3, H, 4*W))
    all_results_color[:, :,:W]= outputs[0][ 0, :, :]
    all_results_color[:, :,W:2*W]= outputs[1][ 0, :, :]
    all_results_color[:, :,2*W:3*W]= outputs[2][ 0, :, :]
    #all_results_color[:,3*W:4*W]= outputs[3][0, :, :]
    #all_results_color[:, :,3*W:4*W] = imgL[0, :, :, :]
    #all_results_color[:, :,3*W:4*W] = disp_L[ 0, :, :]

    im = all_results_color.numpy()
    im = np.transpose(im, (1, 2, 0))
    im= im[:,:,0]

    print(im.shape)
    im = disp2rgb(im/192) * 255

    #im_color = cv.applyColorMap(im*2, cv.COLORMAP_JET)
    cv.imwrite(os.path.join("results/fig", f"out{i}.jpg"),im)
    i+=1

    info_str = ', '.join(['Stage {}={:.4f}'.format(x, D1s[x].avg) for x in range(stages)])
    log.info('Average test 3-Pixel Error = ' + info_str)


def error_estimating(disp, ground_truth, maxdisp=192):
    gt = ground_truth
    mask = gt > 0
    mask = mask * (gt < maxdisp)



    errmap = torch.abs(disp - gt)

    print(errmap)


    err3 = ((errmap[mask] > 3.) & (errmap[mask] / gt[mask] > 0.05)).sum()
    return err3.float() / mask.sum().float()

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = args.lr
    elif epoch <= 400:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
