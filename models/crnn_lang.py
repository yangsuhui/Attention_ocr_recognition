# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import random
import math
import numpy as np

gpu_device = torch.device("cuda")
cpu_device = torch.device("cpu")

GO = 0
EOS_TOKEN = 1              # 结束标志的标签

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
   

class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings=128):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size,bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size+num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_embeddings = num_embeddings
        self.processed_batches = 0

    def forward(self, prev_hidden, feats, cur_embeddings):
        nT = feats.size(0)
        nB = feats.size(1)
        nC = feats.size(2)
        hidden_size = self.hidden_size
        input_size = self.input_size

        feats_proj = self.i2h(feats.view(-1,nC))
        prev_hidden_proj = self.h2h(prev_hidden).view(1,nB, hidden_size).expand(nT, nB, hidden_size).contiguous().view(-1, hidden_size)
        emition = self.score(torch.tanh(feats_proj + prev_hidden_proj).view(-1, hidden_size)).view(nT,nB).transpose(0,1)
        self.processed_batches = self.processed_batches + 1

        if self.processed_batches % 10000 == 0:
            print('processed_batches = %d' %(self.processed_batches))

        alpha = F.softmax(emition) # nB * nT
        if self.processed_batches % 10000 == 0:
            print('emition ', list(emition.data[0]))
            print('alpha ', list(alpha.data[0]))
        context = (feats * alpha.transpose(0,1).contiguous().view(nT,nB,1).expand(nT, nB, nC)).sum(0).squeeze(0) # nB * nC//感觉不应该sum，输出4×256
        context = torch.cat([context, cur_embeddings], 1)
        cur_hidden = self.rnn(context, prev_hidden)
        return cur_hidden, alpha


class DecoderRNN(nn.Module):
    """
        采用RNN进行解码
    """
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))

        return result


class Attentiondecoder(nn.Module):
    """
        采用attention注意力机制，进行解码
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Attentiondecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # calculate the attention weight and weight * encoder_output feature
        embedded = self.embedding(input)         # 前一次的输出进行词嵌入
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)        # 上一次的输出和隐藏状态求出权重, 主要使用一个linear layer从512维到71维，所以只能处理固定宽度的序列
        attn_applied = torch.matmul(attn_weights.unsqueeze(1),
                                 encoder_outputs.permute((1, 0, 2)))      # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256

        output = torch.cat((embedded, attn_applied.squeeze(1) ), 1)       # 上一次的输出和attention feature做一个融合，再加一个linear layer
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)                         # just as sequence to sequence decoder

        output = F.log_softmax(self.out(output[0]), dim=1)          # use log_softmax for nllloss
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


def target_txt_decode(batch_size, text_length, text):
    '''
        对target txt每个字符串的开始加上GO，最后加上EOS，并用最长的字符串做对齐
    return:
        targets: num_steps+1 * batch_size
    '''
    nB = batch_size      # batch

    # 将text分离出来
    num_steps = text_length.data.max()
    num_steps = int(num_steps.cpu().numpy())
    targets = torch.ones(nB, num_steps + 2) * 2                 # 用$符号填充较短的字符串, 在最开始加上GO,结束加上EOS_TOKEN
    targets = targets.long().cuda()        # 用
    start_id = 0
    for i in range(nB):
        targets[i][0] = GO    # 在开始的加上开始标签
        targets[i][1:text_length.data[i] + 1] = text.data[start_id:start_id+text_length.data[i]]       # 是否要加1
        targets[i][text_length.data[i] + 1] = EOS_TOKEN         # 加上结束标签
        start_id = start_id+text_length.data[i]                 # 拆分每个目标的target label，为：batch×最长字符的numel
    targets = Variable(targets.transpose(0, 1).contiguous())
    return targets
    

class CNN(nn.Module):
    '''
        CNN+BiLstm做特征提取
    '''
    def __init__(self, imgH, nc, nh, cfg, mode='1D', dim_in=512):
        super(CNN, self).__init__()

        self.mode = mode
        self.cfg = cfg
        
        if self.mode == '1D':
            assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
                      nn.Conv2d(nc, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 64x16x50
                      nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2), # 128x8x25
                      nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 256x8x25
                      nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 256x4x25
                      nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 512x4x25
                      nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2,2), (2,1), (0,1)), # 512x2x25
                      nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)) # 512x1x25
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nh))
            
        if self.cfg.SEQUENCE.TWO_CONV:   ##finetune.yaml设置的true，dim_in设置的256
            self.seq_encoder = nn.Sequential(nn.Conv2d(dim_in, dim_in, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True), nn.Conv2d(dim_in, 256, 3, padding=1), nn.ReLU(inplace=True))
        else:
            self.seq_encoder = nn.Sequential(nn.Conv2d(dim_in, 256, 3, padding=1), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2, ceil_mode=True))

        self.rescale = nn.Upsample(size=(16, 64), mode='bilinear', align_corners=False)

        ##注意：nn.Embedding输入必须是LongTensor，FloatTensor须通过tensor.long()方法转成LongTensor。
        self.x_onehot = nn.Embedding(32, 32)    ##初始化词向量，32个词，每个词32维度
        self.x_onehot.weight.data = torch.eye(32)   ##one-hot形式，对角线是1，其余位置0
        self.y_onehot = nn.Embedding(8, 8)   ##初始化词向量，8个词，每个词8维度
        self.y_onehot.weight.data = torch.eye(8)

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        #print(conv.size())
        b, c, h, w = conv.size()
        if self.mode == '1D':

            assert h == 1, "the height of conv must be 1"
            conv = conv.squeeze(2)
            conv = conv.permute(2, 0, 1)  # [w, b, c]

            # rnn features calculate
            encoder_outputs = self.rnn(conv)          # seq * batch * n_classes// 25 × batchsize × 256（隐藏节点个数）
            return encoder_outputs
        else:

            ##输入的mask head的特征图大小32*128
            rescale_out = self.rescale(conv)   ##[B,256,16,64]，256是特征图的通道数，16和64对应特征图的高和宽
            seq_decoder_input = self.seq_encoder(rescale_out)  ##[B,256,8,32]
            x_t, y_t = np.meshgrid(np.linspace(0, 31, 32), np.linspace(0, 7, 8))  # (h, w)
            ##x_t和y_t对应特征图的每一个特征点的横纵坐标
            x_t = torch.LongTensor(x_t, device=cpu_device).cuda()   ##x_t是8*32维度
            y_t = torch.LongTensor(y_t, device=cpu_device).cuda()   ##y_t是8*32维度
            ##x_onehot_embedding维度为(B,32,8,32),其中第一个32是每一个字符的编码长度，后面的8,32是x_t坐标对应的8和32
            ##y_onehot_embedding维度意义同x_onehot_embedding，不过具体维度是(B,8,8,32)，因为特征图高度只有8，因此纵坐标编码长度也只是8
            x_onehot_embedding = self.x_onehot(x_t).transpose(0, 2).transpose(1, 2).repeat(seq_decoder_input.size(0),1,1,1)
            y_onehot_embedding = self.y_onehot(y_t).transpose(0, 2).transpose(1, 2).repeat(seq_decoder_input.size(0),1,1,1)
            ##seq_decoder_input_loc是在特征图通道维度上concat横纵坐标的embedding
            seq_decoder_input_loc = torch.cat([seq_decoder_input, x_onehot_embedding, y_onehot_embedding], 1)  ##[B,256+32+8,8,32]
            ##seq_decoder_input_reshape：[8*32,B,256+32+8]
            seq_decoder_input_reshape = seq_decoder_input_loc.view(seq_decoder_input_loc.size(0), seq_decoder_input_loc.size(1), -1).transpose(0, 2).transpose(1, 2)

            return seq_decoder_input_reshape


class decoder(nn.Module):
    '''
        decoder from image features
    '''
    def __init__(self, nh=256, nclass=13, dropout_p=0.1, max_length=71):
        super(decoder, self).__init__()
        self.hidden_size = nh
        self.decoder = Attentiondecoder(nh, nclass, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result


class AttentiondecoderV2(nn.Module):
    """
        采用seq to seq模型，修改注意力权重的计算方式
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttentiondecoderV2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

        # test
        self.vat = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)         # 前一次的输出进行词嵌入  (Batch_size, hidden_size)
        embedded = self.dropout(embedded)   ##对一个Batch中的每一个hidden_size做dropout

        # test
        batch_size = encoder_outputs.shape[1]

        # # 特征融合采用+/concat其实都可以，这里是+，所以维度
        #70(280的宽度4倍下采样) × batchsize × 256（隐藏节点个数）
        alpha = hidden + encoder_outputs        
        alpha = alpha.view(-1, alpha.shape[-1])  ##维度:(70*batch_size,hidden_size)

        # 将encoder_output:batch*seq*features,将features的维度降为1,这里的seq就是图片下采样后的宽度70，features就是每一个字符出的encode
        # 也就是hidden_size
        attn_weights = self.vat( torch.tanh(alpha))    ##维度:(70*batch_size,1)                    
        attn_weights = attn_weights.view(-1, 1, batch_size).permute((2,1,0))  ##维度:(batch_size, 1, 70)
        ##在维度2上进行softmax，也就是seq(70)，一句话的长度上进行softmax
        ##这样得到的就是对一句话每一个字符位置处的概率值，作为attention的权重
        attn_weights = F.softmax(attn_weights, dim=2)   ##维度:(batch_size, 1, 70)

        # attn_weights = F.softmax(
        #     self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)        # 上一次的输出和隐藏状态求出权重

        ##维度：(batch_size,1,hidden_size)
        attn_applied = torch.matmul(attn_weights,
                                 encoder_outputs.permute((1, 0, 2)))      # 矩阵乘法，bmm（8×1×56，8×56×256）=8×1×256
        ##维度: (Batch_size, hidden_size*2)
        #print('embedded.size:',embedded.size())
        #print('attn_applied.squeeze(1).size:',attn_applied.squeeze(1).size())
        output = torch.cat((embedded, attn_applied.squeeze(1) ), 1)       # 上一次的输出和attention feature，做一个线性+GRU
        output = self.attn_combine(output).unsqueeze(0)   ##(1, Batch_size, hidden_size)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)          # 最后输出一个概率 (Batch_size, output_size), output_size上求得log_softmax
        return output, hidden, attn_weights

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))

        return result


class Attn(nn.Module):
    def __init__(self, method, hidden_size, embed_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_size + 32 + 8, hidden_size)    ##(2*256 + 32 +8, 256)
        # self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (B, hidden_size)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (H*W, B, hidden_size)    ##(H*W, B , C)
        :return
            attention energies in shape (B, H*W)
        '''
        max_len = encoder_outputs.size(0)    ##H*W
        this_batch_size = encoder_outputs.size(1)   ##B
        H = hidden.repeat(max_len,1,1).transpose(0,1) # (B, H*W, hidden_size)
        encoder_outputs = encoder_outputs.transpose(0,1) # (B, H*W, hidden_size)  ##(B, H*W, 256+32+8)
        attn_energies = self.score(H,encoder_outputs) # compute attention score (B, H*W)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax (B, 1, H*W)

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # (B, H*W, 2*hidden_size+H+W)->(B, H*W, hidden_size)
        energy = energy.transpose(2,1) # (B, hidden_size, H*W)
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) # (B, 1, hidden_size)
        energy = torch.bmm(v,energy) # (B, 1, H*W)
        return energy.squeeze(1) # (B, H*W)

class BahdanauAttnDecoderRNN(nn.Module):
    ##默认设置的hidden_size:256; embed_size:38,output_size:38;
    # self.seq_decoder = BahdanauAttnDecoderRNN(256, cfg.SEQUENCE.NUM_CHAR, cfg.SEQUENCE.NUM_CHAR, n_layers=1, dropout_p=0.1)
    def __init__(self, hidden_size, embed_size, output_size, n_layers=1, dropout_p=0, bidirectional=False):
        super(BahdanauAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size)    ##初始化词向量，(38,38)
        self.embedding.weight.data = torch.eye(embed_size)        ##one-hot
        # self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(embed_size, hidden_size)   
        self.attn = Attn('concat', hidden_size, embed_size)    
        self.rnn = nn.GRUCell(2 * hidden_size + 32 + 8, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, encoder_outputs):
        '''

         decoder_output, decoder_hidden, decoder_attention = self.seq_decoder(
                        decoder_input, decoder_hidden, seq_decoder_input_reshape)

        :param word_input:
            word input for current time step, in shape (B)  ##[B,1]
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B, hidden_size)  ##[B,256]
        :param encoder_outputs:
            encoder outputs in shape (H*W, B, C)   ##图片CNN后的特征，[8*32,B,256+32+8]
        :return
            decoder output
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded_onehot = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,embed_size)
        word_embedded = self.word_linear(word_embedded_onehot) #(1, B, hidden_size)
        attn_weights = self.attn(last_hidden, encoder_outputs) # (B, 1, H*W)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B, 1, H*W) * (B, H*W, C) = (B,1,C)
        context = context.transpose(0, 1)  # (1,B,C)
        # Combine embedded input word and attended context, run through RNN
        # 2 * hidden_size + W + H: 256 + 256 + 32 + 8 = 552
        rnn_input = torch.cat((word_embedded, context), 2)   ##(1,B,hidden+C)
        last_hidden = last_hidden.view(last_hidden.size(0), -1)  ##(B,256)
        rnn_input = rnn_input.view(word_input.size(0), -1)   ##(B,hidden+C)
        hidden = self.rnn(rnn_input, last_hidden)    ##(B,hidden+c(2*hidden+32+8)),(B,256)
        # if not training:
        #     output = F.softmax(self.out(hidden), dim=1)    ##256->38(B,38)
        # else:
        output = F.log_softmax(self.out(hidden), dim=1)
        # Return final output, hidden state
        # print(output.shape)
        ##维度说明
        ##output:[B,38(char_classes + 结束符)]
        ##hidden:[B,256(hidden size)]
        ##attn_weights:[B, 1, H*W]
        return output, hidden, attn_weights

class SequencePredictor(nn.Module):
    def __init__(self, cfg, nclass):
        super(SequencePredictor, self).__init__()
        self.cfg = cfg
        ##cfg.SEQUENCE.NUM_CHAR默认38
        self.seq_decoder = BahdanauAttnDecoderRNN(256, nclass, nclass, n_layers=1, dropout_p=0.1)

        # for name, param in self.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         # Caffe2 implementation uses MSRAFill, which in fact
        #         # corresponds to kaiming_normal_ in PyTorch
        #         nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

    def forward(self, input, hidden, encoder_outputs):
        return self.seq_decoder(input, hidden, encoder_outputs)


class decoderV2(nn.Module):
    '''
        decoder from image features
    '''

    def __init__(self, nh=256, nclass=13, dropout_p=0.1):
        super(decoderV2, self).__init__()
        self.hidden_size = nh
        self.decoder = AttentiondecoderV2(nh, nclass, dropout_p)

    def forward(self, input, hidden, encoder_outputs):
        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result
