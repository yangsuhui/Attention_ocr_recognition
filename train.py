# coding:utf-8
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import os
import src.utils as utils
import src.dataset as dataset
import time
from src.utils import alphabet
from src.utils import weights_init

from src.defaults import _C as cfg
import src.beam_search as beam_search_sevice

import models.crnn_lang as crnn
print(crnn.__name__)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--trainlist',  default='./data/train_v.txt')
parser.add_argument('--vallist',  default='./data/test_v.txt')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--imgH', type=int, default=420, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=420, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--experiment', default='./expr/attention2dcnn_v', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=10, help='Interval to be displayed')
parser.add_argument('--testInterval', type=int, default=1, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=20, help='Interval to be displayed')
parser.add_argument('--adam', default=True, action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--use_beam_search', default=False, action='store_true', help='Whether to use beam search')
parser.add_argument('--max_width', type=int, default=71, help='the width of the featuremap out from cnn')
parser.add_argument('--mode', type=str, default='2D', help='the mode of attention')
opt = parser.parse_args()
print(opt)

SOS_token = 0              # 开始标志的标签(BOS、SOS)
EOS_TOKEN = 1              # 结束标志的标签
BLANK = 2                  # blank for padding


if opt.experiment is None:
    opt.experiment = 'expr'
os.system('mkdir -p {0}'.format(opt.experiment))        # 创建多级目录

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = None
train_dataset = dataset.listDataset(list_file =opt.trainlist, transform=transform)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))

if opt.mode == '1D':
    test_dataset = dataset.listDataset(list_file =opt.vallist, transform=dataset.resizeNormalize((opt.imgW, opt.imgH)))
else:
    test_dataset = dataset.listDataset(list_file =opt.vallist, transform=dataset.paddingNormalize(opt.imgH, opt.imgW))

nclass = len(alphabet) + 3          # decoder的时候，需要的类别数,3 for SOS,EOS和blank 
print('nclass:',nclass)
cfg.SEQUENCE.NUM_CHAR = nclass
nc = 1

converter = utils.strLabelConverterForAttention(alphabet)
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.NLLLoss()              # 最后的输出要为log_softmax

if opt.mode == '1D':
    encoder = crnn.CNN(opt.imgH, nc, opt.nh, cfg)
    decoder = crnn.decoderV2(opt.nh, nclass, dropout_p=0.1)
else:
    encoder = crnn.CNN(opt.imgH, nc, opt.nh, cfg, mode='2D',dim_in=512)
    # decoder = crnn.decoder(opt.nh, nclass, dropout_p=0.1, max_length=opt.max_width)        # max_length:w/4,为encoder特征提取之后宽度方向上的序列长度
    #decoder = crnn.decoderV2(opt.nh, nclass, dropout_p=0.1)        # For prediction of an indefinite long sequence
    decoder = crnn.SequencePredictor(cfg, nclass)
    
encoder.apply(weights_init)
decoder.apply(weights_init)
# continue training or use the pretrained model to initial the parameters of the encoder and decoder
if opt.encoder:
    print('loading pretrained encoder model from %s' % opt.encoder)
    encoder.load_state_dict(torch.load(opt.encoder))
if opt.decoder:
    print('loading pretrained decoder model from %s' % opt.decoder)
    decoder.load_state_dict(torch.load(opt.decoder))
print(encoder)
print(decoder)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.LongTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    encoder.cuda()
    decoder.cuda()
    # encoder = torch.nn.DataParallel(encoder, device_ids=range(opt.ngpu))
    # decoder = torch.nn.DataParallel(decoder, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

'''
1、自己设置lr随着epoch的变化
lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [200,400]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

# optimizer = torch.optim.SGD(models.parameters(), lr=config.lr, momentum=0.99)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
if config.checkpoint != '' and not config.restart_training:
    start_epoch = load_checkpoint(config.checkpoint, model, logger, device, optimizer)
    start_epoch += 1
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma,
                                                        last_epoch=start_epoch)
else:
    start_epoch = config.start_epoch
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.lr_decay_step, gamma=config.lr_gamma)

# learning rate的warming up操作
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

lr = adjust_learning_rate(optimizer, epoch)

2、设置torch自带的lr的变化

  OPTIMIZER: 'adam'
  LR: 0.0001
  WD: 0.0
  LR_STEP: [10, 20]
  LR_FACTOR: 0.1
  MOMENTUM: 0.0
  NESTEROV: False
  RMSPROP_ALPHA:
  RMSPROP_CENTERED:

import torch.optim as optim
def get_optimizer(config, model):

    optimizer = None

    if config.TRAIN.OPTIMIZER == "sgd":
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            nesterov=config.TRAIN.NESTEROV
        )
    elif config.TRAIN.OPTIMIZER == "adam":
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
        )
    elif config.TRAIN.OPTIMIZER == "rmsprop":
        optimizer = optim.RMSprop(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.TRAIN.LR,
            momentum=config.TRAIN.MOMENTUM,
            weight_decay=config.TRAIN.WD,
            # alpha=config.TRAIN.RMSPROP_ALPHA,
            # centered=config.TRAIN.RMSPROP_CENTERED
        )

    return optimizer

    last_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = utils.get_optimizer(config, model)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch - 1
        )
    
    lr_scheduler.step()
    lr = lr_scheduler.get_lr()[0]

'''

'''
统计模型参数量等信息
def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))
'''

# setup optimizer
if opt.adam:
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=opt.lr,
                        betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(encoder.parameters(), lr=opt.lr)
else:
    encoder_optimizer = optim.RMSprop(encoder.parameters(), lr=opt.lr)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=opt.lr)


def val(cfg, encoder, decoder, criterion, batchsize, dataset, teach_forcing=False, max_iter=100, use_beam_search=False, mode='1D'):
    
    print('Start val')

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batchsize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    # max_iter = len(data_loader) - 1
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        b = cpu_images.size(0)
        utils.loadData(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        ##因为batch_size是1，所以这里的n_total是单个text的长度+EOS停止位的预测
        n_total += len(cpu_texts[0]) + 1                       # 还要准确预测出EOS停止位

        decoded_words = []
        decoded_label = []
        #decoder_attentions = torch.zeros(len(cpu_texts[0]) + 1, opt.max_width)
        #encoder_outputs = encoder(image)            # cnn+biLstm做特征提取
        target_variable = target_variable.cuda()
        #decoder_input = target_variable[0].cuda()   # 初始化decoder的开始,从0开始输出
        #decoder_hidden = decoder.initHidden(b).cuda()
        loss = 0.0
        encoder_outputs = encoder(image)               # cnn+biLstm做特征提取  维度：70(280的宽度4倍下采样) × batchsize × 256（隐藏节点个数）

        if mode=='1D':                    
            decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出 (batch 个label的开始SOS:0), 维度batch size
            decoder_hidden = decoder.initHidden(b).cuda()  #维度:(1, batch_size, self.hidden_size)
        else:
            bos_onehot = np.zeros((encoder_outputs.size(1), 1), dtype=np.int32)  ##[B,1]
            bos_onehot[:, 0] = cfg.SEQUENCE.BOS_TOKEN  ##0
            decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)   ##列表,[B,1],数字0
            decoder_hidden = torch.zeros((encoder_outputs.size(1), 256), device=gpu_device)  ##[B,256]

        ##测试函数val
        ##inference时words存储的是batch的word，最大长度32；
        ##decoded_scores是batch的每一个字符的置信度
        ##detailed_decoded_scores只有采用beam_search时才有数，不然就是空列表，具体的存放的是指定topk的置信度，即预测每一个字符时保存topk个当前预测字符的置信度
        ##例如一张图片预测出10个框(batch是10)，每个batch中在预测具体的字符，这样words就是10个，decoded_scores是10个words对应的组成字符的置信度
        ##每一个word组成的字符置信度是一个列表

        char_scores = []
        detailed_char_scores = []

        #if not teach_forcing:
        if not use_beam_search:
            # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
            for di in range(1, target_variable.shape[0]):  # 最大字符串的长度
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
                loss_avg.add(loss)
                #decoder_attentions[di-1] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                char_scores.append(topv.item())   ##预测的topk(1)对应的字符的置信度(经过softmax之后)
                ni = topi.squeeze(1)
                decoder_input = ni
                #if ni.item() == EOS_TOKEN:
                if ni == EOS_TOKEN:
                    decoded_words.append('<EOS>')
                    decoded_label.append(EOS_TOKEN)
                    break
                else:
                    #decoded_words.append(converter.decode(ni.item()))
                    decoded_words.append(converter.decode(ni))
                    decoded_label.append(ni)
                    
        else:

            #top_seqs = decoder.beam_search(encoder_outputs, decoder_hidden, beam_size=6, max_len=cfg.SEQUENCE.MAX_LENGTH)
            top_seqs = beam_search_sevice.beam_search(cfg, mode, decoder, encoder_outputs, decoder_hidden, beam_size=6, max_len=cfg.SEQUENCE.MAX_LENGTH )
            top_seq = top_seqs[0]
            for character in top_seq[1:]:
                character_index = character[0]
                if character_index == EOS_TOKEN:
                    char_scores.append(character[1])
                    detailed_char_scores.append(character[2])
                    #decoded_words.append('<EOS>')
                    decoded_label.append(EOS_TOKEN)
                    break
                else:
                    if character_index == 0:
                        decoded_words.append('~')
                        char_scores.append(0.)
                        decoded_label.append(0)
                    else:
                        decoded_words.append(converter.decode(character_index))
                        decoded_label.append(character_index)
                        char_scores.append(character[1])
                        detailed_char_scores.append(character[2])

        # 计算正确个数
        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        #if i % 100 == 0:                 # 每100次输出一次
        if i % 2 == 0:                 # 每100次输出一次
            texts = cpu_texts[0]
            print('pred:%-20s, gt: %-20s' % (decoded_words, texts))

    accuracy = n_correct / float(n_total)   
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, teach_forcing_prob=1, mode='1D'):
    '''
        target_label:采用后处理的方式，进行编码和对齐，以便进行batch训练
    '''
    data = train_iter.next()
    cpu_images, cpu_texts = data
    b = cpu_images.size(0)  ##batch size大小
    target_variable = converter.encode(cpu_texts) ##max_length × batch_size
    target_variable = target_variable.cuda()
    utils.loadData(image, cpu_images)
    encoder_outputs = encoder(image)               # cnn+biLstm做特征提取  维度：70(280的宽度4倍下采样) × batchsize × 256（隐藏节点个数）

    if mode=='1D':                
        decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出 (batch 个label的开始SOS:0), 维度batch size
        decoder_hidden = decoder.initHidden(b).cuda()  #维度:(1, batch_size, self.hidden_size)
    else:
        #print('-----------:',encoder_outputs.size(1))
        bos_onehot = np.zeros((encoder_outputs.size(1), 1), dtype=np.int32)  ##[B,1]
        bos_onehot[:, 0] = cfg.SEQUENCE.BOS_TOKEN  ##0
        decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)   ##列表,[B,1],数字0
        decoder_hidden = torch.zeros((encoder_outputs.size(1), 256), device=gpu_device)  ##[B,256]
        ##TEACHER_FORCE_RATIO:1
        #use_teacher_forcing = True if random.random() < self.cfg.SEQUENCE.TEACHER_FORCE_RATIO else False
        #target_length = decoder_targets.size(1)  ##32,每一个roi mask区域的target label的max_length,统一到一个max_length长度

    # if use_teacher_forcing:
    #     # Teacher forcing: Feed the target as the next input   
    #     ##维度说明
    #     ##output:[B,38(char_classes + 结束符)]
    #     ##hidden:[B,256(hidden size)]
    #     ##attn_weights:[B, 1, H*W]           
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
    #             decoder_input, decoder_hidden, seq_decoder_input_reshape)
    #         if di == 0:
    #             loss_seq_decoder = self.criterion_seq_decoder(decoder_output, word_targets[:,di])
    #         else:
    #             loss_seq_decoder += self.criterion_seq_decoder(decoder_output, word_targets[:,di])
    #         decoder_input = decoder_targets[:, di] # Teacher forcing
    # else:
    #     # Without teacher forcing: use its own predictions as the next input
    #     for di in range(target_length):
    #         decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
    #             decoder_input, decoder_hidden, seq_decoder_input_reshape)
    #         topv, topi = decoder_output.topk(1)   ##topv是取得top1对应的值，topi是对应的值的索引，维度[B]
    #         decoder_input = topi.squeeze(1).detach()  # detach from history as input  维度[B,1]
    #         if di == 0:
    #             loss_seq_decoder = self.criterion_seq_decoder(decoder_output, word_targets[:,di])
    #         else:
    #             loss_seq_decoder += self.criterion_seq_decoder(decoder_output, word_targets[:,di])
    # loss_seq_decoder = loss_seq_decoder.sum() / loss_seq_decoder.size(0)
    # loss_seq_decoder = 0.2 * loss_seq_decoder
    

    loss = 0.0
    teach_forcing = True if random.random() > teach_forcing_prob else False
    #loss = decoder(encoder_outputs, target_variable, target_variable, False, teach_forcing)
    if teach_forcing:
        # 教师强制：将目标label作为下一个输入
        for di in range(1, target_variable.shape[0]):           # 最大字符串的长度
            #decoder_output: (Batch_size, output_size)
            #decoder_hidden: (1, batch_size, self.hidden_size)
            #decoder_attention(attention的权重): (batch_size, 1, 70)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])          # 每次预测一个字符
            decoder_input = target_variable[di]  # Teacher forcing/前一次的输出
    else:
        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])  # 每次预测一个字符
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze()
            decoder_input = ni

    encoder.zero_grad()
    decoder.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


if __name__ == '__main__':
    t0 = time.time()
    #val(cfg, encoder, decoder, criterion, 1, dataset=test_dataset, teach_forcing=False, use_beam_search=opt.use_beam_search, mode=opt.mode) 
    for epoch in range(opt.niter):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader)-1:
            for e, d in zip(encoder.parameters(), decoder.parameters()):
                e.requires_grad = True
                d.requires_grad = True
            encoder.train()
            decoder.train()
            cost = trainBatch(encoder, decoder, criterion, encoder_optimizer, 
                              decoder_optimizer, teach_forcing_prob=opt.teaching_forcing_prob, mode=opt.mode)
            loss_avg.add(cost)
            #print('loss:',loss_avg.val())
            i += 1

            if i % opt.displayInterval == 0:
                print('[%d/%d][%d/%d] Loss: %f' %
                    (epoch, opt.niter, i, len(train_loader), loss_avg.val()), end=' ')
                loss_avg.reset()
                t1 = time.time()
                print('time elapsed %d' % (t1-t0))
                t0 = time.time()

        if epoch % opt.testInterval == 0:
            val(cfg, encoder, decoder, criterion, 1, dataset=test_dataset, teach_forcing=False, use_beam_search=opt.use_beam_search, mode=opt.mode)            # batchsize:1
        # do checkpointing
        if epoch % opt.saveInterval == 0:
            torch.save(
                encoder.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch))
            torch.save(
                decoder.state_dict(), '{0}/decoder_{1}.pth'.format(opt.experiment, epoch))