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

def reduce_mul(l):
    out = 1.0
    for x in l:
        out *= x
    return out

def check_all_done(seqs):
    for seq in seqs:
        if not seq[-1]:
            return False
    return True

def beam_search_step(cfg, mode, seq_decoder, encoder_context, top_seqs, k):       
    all_seqs = []
    for seq in top_seqs:
        seq_score = reduce_mul([_score for _,_score,_,_ in seq])   ##初始值是1
        #if seq[-1][0] == cfg.SEQUENCE.NUM_CHAR - 1:
        if seq[-1][0] == cfg.SEQUENCE.EOS_TOKEN:
            all_seqs.append((seq, seq_score, seq[-1][2], True))
            continue
        decoder_hidden = seq[-1][-1][0]
        if mode == '1D':
            #decoder_input = torch.zeros(1).long().cuda()
            decoder_input = torch.ones(1).long().cuda()
            decoder_input *= seq[-1][0]
        else:                
            onehot = np.zeros((1, 1), dtype=np.int32)
            onehot[:, 0] = seq[-1][0]
            decoder_input = torch.tensor(onehot.tolist(), device=gpu_device)

        decoder_output, decoder_hidden, decoder_attention = seq_decoder(
                decoder_input, decoder_hidden, encoder_context)
        #RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        detailed_char_scores = decoder_output.cpu().detach().numpy()
        # print(decoder_output.shape)
        # decoder_output.data[:,1:]除掉第一个位置0表示的BOS
        #scores维度: (B,k)
        #candidates维度: (B,k)
        decoder_output = torch.exp(decoder_output)

        scores, candidates = decoder_output.data[:,1:].topk(k)   
        for i in range(k):
            character_score = scores[:, i]
            character_index = candidates[:, i]
            score = seq_score * character_score.item()
            char_score = seq_score * detailed_char_scores
            rs_seq = seq + [(character_index.item() + 1, character_score.item(), char_score, [decoder_hidden])]
            #done = (character_index.item() + 1 == cfg.SEQUENCE.NUM_CHAR - 1) 
            done = (character_index.item() + 1 == cfg.SEQUENCE.EOS_TOKEN)     
            all_seqs.append((rs_seq, score, char_score, done))           
    all_seqs = sorted(all_seqs, key = lambda seq: seq[1], reverse=True)   ##从大到小排列
    topk_seqs = [seq for seq,_,_,_ in all_seqs[:k]]
    all_done = check_all_done(all_seqs[:k])
    return topk_seqs, all_done

def beam_search(cfg, mode, seq_decoder, encoder_context, decoder_hidden, beam_size=6, max_len=32):
    '''
    top_seqs = self.beam_search(seq_decoder_input_reshape[:,batch_index:batch_index+1,:], decoder_hidden, beam_size=6, max_len=self.cfg.SEQUENCE.MAX_LENGTH)
    encoder_context是图片的编码特征和位置编码的concat，也就是seq_decoder_input_reshape[:,batch_index:batch_index+1,:]
    '''
    #char_score = np.zeros(38,)
    #char_score = np.zeros(5994,)
    char_score = np.zeros(cfg.SEQUENCE.NUM_CHAR,)
    ##(character_index.item() + 1, character_score.item(), char_score, [decoder_hidden])
    top_seqs = [[(cfg.SEQUENCE.BOS_TOKEN, 1.0, char_score, [decoder_hidden])]] 
    #loop
    for _ in range(max_len):        
        top_seqs, all_done = beam_search_step(cfg, mode, seq_decoder, encoder_context, top_seqs, beam_size)
        if all_done:            
            break        
    return top_seqs