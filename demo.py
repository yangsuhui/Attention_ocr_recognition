# coding:utf-8

'''
March 2019 by Chen Jun
https://github.com/chenjun2hao/Attention_ocr.pytorch

'''
import argparse
import torch
from torch.autograd import Variable
import src.utils as utils
import src.dataset as dataset
from PIL import Image
from src.utils import alphabet
import models.crnn_lang as crnn
from src.defaults import _C as cfg
import numpy as np
import os
import src.beam_search as beam_search_sevice

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='./expr/attention2dcnn/encoder_490.pth', help="path to encoder to test")
parser.add_argument('--decoder', type=str, default='./expr/attention2dcnn/decoder_490.pth', help='path to decoder to test')
parser.add_argument('--imgH', type=int, default=320, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=320, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--img_path', type=str, default='./test_img/20441531_4212871437.jpg', help='which img to test')
parser.add_argument('--use_gpu', default=True, action='store_true', help='Whether to use cuda')
parser.add_argument('--use_beam_search', default=False, action='store_true', help='Whether to use beam search')
parser.add_argument('--max_length', type=int, default=15, help='the width of the length of sentence')
parser.add_argument('--EOS_TOKEN', type=int, default=1, help='the id of EOS')
parser.add_argument('--dim_in', type=int, default=512, help='the dim in')
parser.add_argument('--mode', type=str, default='2D', help='the mode of attention')
opt = parser.parse_args()

#use_gpu = True
# encoder_path = './expr/attentioncnn/encoder_10.pth'
# decoder_path = './expr/attentioncnn/decoder_10.pth'
# img_path = './test_img/20441531_4212871437.jpg'
#max_length = 15                          # 最长字符串的长度
#EOS_TOKEN = 1

nclass = len(alphabet) + 3
cfg.SEQUENCE.NUM_CHAR = nclass
nc = 1
if opt.mode == '1D':
    encoder = crnn.CNN(32, 1, 256, cfg)          # 编码器
    decoder = crnn.decoderV2(256, nclass)     # seq to seq的解码器, nclass在decoder中还加了2
    #decoder = crnn.decoderV2(256, nclass - 2)
else:
    encoder = crnn.CNN(opt.imgH, nc, opt.nh, cfg, mode=opt.mode,dim_in=opt.dim_in)
    decoder = crnn.SequencePredictor(cfg, nclass)


if opt.encoder and opt.decoder:
    print('loading pretrained models ......')

    encoder_dict = encoder.state_dict()
    trained_encoder_dict = torch.load(opt.encoder)
    state_dict = {k:v for k,v in trained_encoder_dict.items() if k in encoder_dict.keys()}
    encoder_dict.update(state_dict)
    encoder.load_state_dict(encoder_dict)

    decoder_dict = decoder.state_dict()
    trained_decoder_dict = torch.load(opt.decoder)
    state_dict = {k:v for k,v in trained_decoder_dict.items() if k in decoder_dict.keys()}
    decoder_dict.update(state_dict)
    decoder.load_state_dict(decoder_dict)    
    # encoder.load_state_dict(torch.load(opt.encoder))
    # decoder.load_state_dict(torch.load(opt.decoder))
if torch.cuda.is_available() and opt.use_gpu:
    encoder = encoder.cuda()
    decoder = decoder.cuda()


converter = utils.strLabelConverterForAttention(alphabet)

if opt.mode == '1D':
#transformer = dataset.resizeNormalize((280, 32))
    transformer = dataset.resizeNormalize((opt.imgW, opt.imgH))
    image = Image.open(opt.img_path).convert('L')
else:
    transformer = dataset.paddingNormalize(opt.imgH, opt.imgW)
    #image = Image.open(opt.img_path).convert('L')
    image = Image.open(opt.img_path)
image = transformer(image)
if torch.cuda.is_available() and opt.use_gpu:
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

encoder.eval()
decoder.eval()

encoder_out = encoder(image)

if opt.mode == '1D':
    #encoder_outputs = encoder(image)               # cnn+biLstm做特征提取  维度：70(280的宽度4倍下采样) × batchsize × 256（隐藏节点个数）        
    #decoder_input = target_variable[0].cuda()      # 初始化decoder的开始,从0开始输出 (batch 个label的开始SOS:0), 维度batch size
    decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
    decoder_hidden = decoder.initHidden(1)  #维度:(1, batch_size, self.hidden_size)
else:
    #encoder_outputs = encoder(image)
    bos_onehot = np.zeros((encoder_out.size(1), 1), dtype=np.int32)  ##[B,1]
    bos_onehot[:, 0] = cfg.SEQUENCE.BOS_TOKEN  ##0
    decoder_input = torch.tensor(bos_onehot.tolist())
    decoder_hidden = torch.zeros((encoder_out.size(1), 256))
    #decoder_input = torch.tensor(bos_onehot.tolist(), device=gpu_device)   ##列表,[B,1],数字0
    #decoder_hidden = torch.zeros((encoder_outputs.size(1), 256), device=gpu_device)  ##[B,256]

decoded_words = []
prob = 1.0
#decoder_attentions = torch.zeros(opt.max_length, 71)
# decoder_input = torch.zeros(1).long()      # 初始化decoder的开始,从0开始输出
# decoder_hidden = decoder.initHidden(1)
if torch.cuda.is_available() and opt.use_gpu:
    decoder_input = decoder_input.cuda()
    decoder_hidden = decoder_hidden.cuda()

#loss = 0.0
# 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
##测试函数val
##inference时words存储的是batch的word，最大长度32；
##decoded_scores是batch的每一个字符的置信度
##detailed_decoded_scores只有采用beam_search时才有数，不然就是空列表，具体的存放的是指定topk的置信度，即预测每一个字符时保存topk个当前预测字符的置信度
##例如一张图片预测出10个框(batch是10)，每个batch中在预测具体的字符，这样words就是10个，decoded_scores是10个words对应的组成字符的置信度
##每一个word组成的字符置信度是一个列表

char_scores = []
detailed_char_scores = []

#if not teach_forcing:
if not opt.use_beam_search:
    # 预测的时候采用非强制策略，将前一次的输出，作为下一次的输入，直到标签为EOS_TOKEN时停止
    for di in range(opt.max_length):  # 最大字符串的长度
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_out)
        probs = torch.exp(decoder_output)
        #decoder_attentions[di-1] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi.squeeze(1)
        decoder_input = ni
        prob *= probs[:, ni]
        char_scores.append(topv.item())   ##预测的topk(1)对应的字符的置信度(经过softmax之后)

        if ni == opt.EOS_TOKEN:
            # decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(converter.decode(ni))
        #if ni.item() == EOS_TOKEN:
    prob = prob.item()
            
else:

    #top_seqs = decoder.beam_search(encoder_out, decoder_hidden, beam_size=6, max_len=cfg.SEQUENCE.MAX_LENGTH)
    top_seqs = beam_search_sevice.beam_search(cfg, opt.mode, decoder, encoder_out, decoder_hidden, beam_size=6, max_len=cfg.SEQUENCE.MAX_LENGTH )
    top_seq = top_seqs[0]
    for character in top_seq[1:]:
        character_index = character[0]
        if character_index == opt.EOS_TOKEN:
            char_scores.append(character[1])
            detailed_char_scores.append(character[2])
            #decoded_words.append('<EOS>')
            break
        else:
            if character_index == 0:
                decoded_words.append('~')
                char_scores.append(1.)
            else:
                decoded_words.append(converter.decode(character_index))
                char_scores.append(character[1])
                detailed_char_scores.append(character[2])
    for i in range(len(char_scores)):
        #print(char_scores[i])
        #print('\n')
        prob*=char_scores[i]

words = ''.join(decoded_words)
print('predict_str:%-20s => prob:%-20s' % (words, prob))

