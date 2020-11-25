import sys
import click
import numpy as np
import PIL.Image as PILImage
import os
import PIL
from matplotlib import transforms
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.image as mpimg  # mpimg 用于读取图片
import matplotlib.pyplot as plt  # plt 用于显示图片

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

def getCombineArray(att_w, img_path):
    #visualize_attention(greedy search时)
    
    att_w = (1 - (att_w.reshape([32,8]) > 0.05) * 1.0) * 255
    #att_w = (1 - (att_w.reshape([71]) > 0.05) * 1.0) * 255

    inp_image = PILImage.open(img_path).convert('L')
    out_image = PILImage.fromarray(att_w).resize(inp_image.size, PILImage.NEAREST)
    #out_image = out_image.convert('RGB')

    # print('shape:',out_image.size,inp_image.size)

    combine = PILImage.blend(inp_image.convert('RGBA'), out_image.convert('RGBA'), 0.5)
    return combine


def rainbow_text(x, y, strings, colors, ax=None, **kw):
    """
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

    The text will get added to the ``ax`` axes, if provided, otherwise the
    currently active axes will be used.
    """
    if ax is None:
        ax = plt.gca()
    t = ax.transData
    canvas = ax.figure.canvas

    # # horizontal version
    for s, c in zip(strings, colors):
        text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        text.draw(canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(
            text.get_transform(), x=ex.width, units='dots')

    # for s, c in zip(strings, colors):
    #     text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
    #     text.draw(canvas.get_renderer())
    #     ex = text.get_window_extent()
    #     t = transforms.offset_copy(
    #         text.get_transform(), y=ex.height, units='dots')

#vis_attention_gif(atte_w, atte_h, path_to_save_attention, opt.img_path, decoder_attentions, decoded_words, full_latex=False)
def vis_attention_gif(att_w, att_h, path_to_save_attention, img_path, hyps, decoded_words, prob, full_latex=False):

    decoded_attentions = hyps
    (img_w, img_h) = PILImage.open(img_path).convert('L').size

    # img, img_w, img_h = readImageAndShape(img_path)
    # att_w, att_h = getWH(img_w, img_h)
    #results_symbols = hyps[0].split(" ")
    results_symbols = decoded_words
    results_symbols_count = len(results_symbols)
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)
    # fig.set_figwidth(25)
    # fig.set_figheight(6)
    fig.set_figwidth(25)
    fig.set_figheight(8)
    # 询问图形在屏幕上的大小和DPI（每英寸点数）
    # 注意当把图形保存为文件时，需要为此单独再提供一个DPI
    print('fig size: {0} DPI, size in inches {1}'.format(fig.get_dpi(), fig.get_size_inches()))
    print('processing...')

    def update(i):
        '''
        在这里绘制动画帧
        args:
        i : (int) range [0, ?)
        return:
        (tuple) 以元组形式返回这一帧需要重新绘制的物体
        '''
        # 1. 更新标题
        results_symbols_colors = ['green']*results_symbols_count
        if i < results_symbols_count:
            results_symbols_colors[i] = "red"
        symbol_count = results_symbols_count if full_latex else i+1
        rainbow_text(-400, img_h+50, results_symbols[:symbol_count], results_symbols_colors[:symbol_count], ax, size=30)
        #rainbow_text(img_w+30, 0, results_symbols[:symbol_count], results_symbols_colors[:symbol_count], ax, size=36)
        # 2. 更新图片
        attentionVector = decoded_attentions[i]
        combine = getCombineArray(attentionVector, img_path)
        #combine = getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h)

        ax.imshow(np.asarray(combine))
        # 3. 以元组形式返回这一帧需要重新绘制的物体
        return ax

    # 会为每一帧调用Update函数
    # 这里FunAnimation设置一个10帧动画，每帧间隔200ms
    plt.title("Visualize Attention over Image: {}".format(prob), fontsize=30)
    anim = FuncAnimation(fig, update, frames=np.arange(0, results_symbols_count), interval=400)
    anim.save(path_to_save_attention+'_visualization_results.gif', dpi=80, writer='imagemagick')
    print("finish!")



