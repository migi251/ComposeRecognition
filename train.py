import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TransformerLabelConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
import modules.transformer_component.Constants as Constants
from Optim import ScheduledOptim
from iotools import save_checkpoint, check_isfile
import pickle
from functools import partial
from tqdm import tqdm
import torch.nn.functional as F


def count_num_param(model):
    num_param = sum(p.numel() for p in model.parameters()) / 1e+06

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
        # we ignore the classifier because it is unused at test time
        num_param -= sum(p.numel()
                         for p in model.classifier.parameters()) / 1e+06
    return num_param


def transformer_loss(pred, gold, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
    else:
        loss = F.cross_entropy(
            pred, gold, ignore_index=Constants.PAD, reduction='mean')

    return loss


def train(opt):
    """ dataset preparation """
    opt.select_data = opt.select_data.split('-')
    opt.batch_ratio = opt.batch_ratio.split('-')
    train_dataset = Batch_Balanced_Dataset(opt)

    AlignCollate_valid = AlignCollate(
        imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size,
        # 'True' to check training progress with validation function.
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)

    """ model configuration """
    if 'Transformer' in opt.SequenceModeling:
        converter = TransformerLabelConverter(opt.character)
    elif 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    """ setup loss """
    if 'Transformer' in opt.SequenceModeling:
        criterion = transformer_loss
    elif 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).cuda()
    else:
        # ignore [GO] token = ignore index 0
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters,
                               lr=opt.lr, betas=(opt.beta1, 0.999))
    elif 'Transformer' in opt.SequenceModeling and opt.use_scheduled_optim:
        optimizer = optim.Adam(filtered_parameters,
                               betas=(0.9, 0.98), eps=1e-09)
        optimizer_schedule = ScheduledOptim(optimizer,
                                            opt.d_model, opt.n_warmup_steps)
    else:
        optimizer = optim.Adadelta(
            filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    pickle.load = partial(pickle.load, encoding="latin1")
    pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    if opt.load_weights != '' and check_isfile(opt.load_weights):
        # load pretrained weights but ignore layers that don't match in size
        checkpoint = torch.load(opt.load_weights, pickle_module=pickle)
        if type(checkpoint) == dict:
            pretrain_dict = checkpoint['state_dict']
        else:
            pretrain_dict = checkpoint
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items(
        ) if k in model_dict and model_dict[k].size() == v.size()}
        model_dict.update(pretrain_dict)
        model.load_state_dict(model_dict)
        print("Loaded pretrained weights from '{}'".format(opt.load_weights))
        del checkpoint
        torch.cuda.empty_cache()
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        checkpoint = torch.load(opt.continue_model)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])
        start_iter = checkpoint['step'] + 1
        print('continue to train start_iter: ', start_iter)
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        if 'best_accuracy' in checkpoint.keys():
            best_accuracy = checkpoint['best_accuracy']
        if 'best_norm_ED' in checkpoint.keys():
            best_norm_ED = checkpoint['best_norm_ED']
        del checkpoint
        torch.cuda.empty_cache()
    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    print("Model size:", count_num_param(model), 'M')
    if 'Transformer' in opt.SequenceModeling and opt.use_scheduled_optim:
        optimizer.n_current_steps = start_iter

    for i in tqdm(range(start_iter, opt.num_iter)):
        for p in model.parameters():
            p.requires_grad = True

        cpu_images, cpu_texts = train_dataset.get_batch()
        image = cpu_images.cuda()
        if 'Transformer' in opt.SequenceModeling:
            text, length, text_pos = converter.encode(
                cpu_texts, opt.batch_max_length)
        elif 'CTC' in opt.Prediction:
            text, length = converter.encode(cpu_texts)
        else:
            text, length = converter.encode(cpu_texts, opt.batch_max_length)
        batch_size = image.size(0)

        if 'Transformer' in opt.SequenceModeling:
            preds = model(image, text, tgt_pos=text_pos)
            target = text[:, 1:]  # without <s> Symbol
            cost = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        elif 'CTC' in opt.Prediction:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)  # to use CTCLoss format
            cost = criterion(preds, text, preds_size, length)
        else:
            preds = model(image, text)
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(
                preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()

        if 'Transformer' in opt.SequenceModeling and opt.use_scheduled_optim:
            optimizer_schedule.step_and_update_lr()
        elif 'Transformer' in opt.SequenceModeling:
            optimizer.step()
        else:
            # gradient clipping with 5 (Default)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

        loss_avg.add(cost)

        # validation part
        if i > 0 and (i+1) % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            print(
                f'[{i+1}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                log.write(
                    f'[{i+1}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
                loss_avg.reset()

                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, preds, gts, infer_time, length_of_data = validation(
                        model, criterion, valid_loader, converter, opt)
                model.train()

                for pred, gt in zip(preds[:5], gts[:5]):
                    if 'Transformer' in opt.SequenceModeling:
                        pred = pred[:pred.find('</s>')]
                        gt = gt[:gt.find('</s>')]
                    elif 'Attn' in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                        gt = gt[:gt.find('[s]')]
                    
                    print(f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                    log.write(
                        f'{pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                valid_log = f'[{i+1}/{opt.num_iter}] valid loss: {valid_loss:0.5f}'
                valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                print(valid_log)
                log.write(valid_log + '\n')

                # keep best accuracy model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    state_dict = model.module.state_dict()
                    save_checkpoint({'best_accuracy': best_accuracy,
                                     'state_dict': state_dict,
                                     }, False, f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED < best_norm_ED:
                    best_norm_ED = current_norm_ED
                    state_dict = model.module.state_dict()
                    save_checkpoint({'best_norm_ED': best_norm_ED,
                                     'state_dict': state_dict,
                                     }, False, f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                    # torch.save(
                    #     model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}'
                print(best_model_log)
                log.write(best_model_log + '\n')

        # save model per 1000 iter.
        if (i + 1) % 1000 == 0:
            state_dict = model.module.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            save_checkpoint({'state_dict': state_dict,
                             'optimizer': optimizer_state_dict,
                             'step': i,
                             'best_accuracy': best_accuracy,
                             'best_norm_ED': best_norm_ED,
                             }, False, f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name',
                        help='Where to store logs and models')
    parser.add_argument('--train_data', default='data_lmdb_release/training',
                        help='path to training dataset')
    parser.add_argument('--valid_data', default='data_lmdb_release/validation',
                        help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int,
                        default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int,
                        default=196, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000,
                        help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=1000,
                        help='Interval between each validation')
    parser.add_argument('--continue_model', default='',
                        help="path to model to continue training")
    parser.add_argument('--load_weights', default='',
                        help="path to model to load pretrain")
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--lr', type=float, default=1,
                        help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95,
                        help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8,
                        help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping value. default=5')
    """ Data processing """
    parser.add_argument('--select_data', type=str, default='MJ-ST',
                        help='select training data (default is MJ-ST, which means MJ and ST used as training data)')
    parser.add_argument('--batch_ratio', type=str, default='0.5-0.5',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int,
                        default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32,
                        help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100,
                        help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true',
                        help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true',
                        help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str,
                        required=False, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=False,
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str,
                        required=False, help='SequenceModeling stage. None|BiLSTM|Transformer')
    parser.add_argument('--Prediction', type=str,
                        required=False, help='Prediction stage. None|CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20,
                        help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1,
                        help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='the size of the LSTM hidden state')
    """ Transformer """
    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_inner_hid', type=int, default=256)
    parser.add_argument('-d_k', type=int, default=64)
    parser.add_argument('-d_v', type=int, default=64)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers_enc', type=int, default=6)
    parser.add_argument('-n_layers_dec', type=int, default=6)
    parser.add_argument('-n_warmup_steps', type=int, default=16000)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-use_scheduled_optim', action='store_true')

    opt = parser.parse_args()

    opt.Transformation = 'None'
    opt.FeatureExtraction = 'RCNN'
    opt.SequenceModeling = 'Transformer'
    opt.Prediction = 'None'
    opt.select_data = 'gene_data'
    opt.valid_data = 'data_lmdb_release/validation_2'
    opt.batch_ratio = '1'
    opt.batch_size = 64
    opt.imgW = 100
    opt.valInterval = 10
    # opt.valid_data = 'data_lmdb_release/validation_2'
    # opt.continue_model = './saved_models/None-ResNet-None-Transformer-Seed1111/iter_300.pth'
    # opt.load_weights = './saved_models/None-ResNet-None-Transformer-Seed1111/best_accuracy_14000.pth'
    opt.experiment_name = 'demo'

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        # same with ASTER setting (use 94 char).
        opt.character = string.printable[:-6]

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
