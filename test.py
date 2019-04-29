from __future__ import print_function

import time
import shutil
import glob
import re
import os
import shutil
import sys
from PIL import Image
import cv2
import numpy as np
import difflib
import Levenshtein
import tensorflow as tf
from tensorflow.python.platform import gfile
from torch.autograd import Variable

from crnn import dataset
from crnn import keys_crnn
from crnn.models import crnn
import torch.utils.data
from crnn import util

path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
# print(path)
sys.path.append(path)
sys.path.append(os.getcwd())

from lib.fast_rcnn.config import cfg, cfg_from_file
from lib.fast_rcnn.test import _get_blobs
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg
from lib.rpn_msr.proposal_layer_tf import proposal_layer


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f


# input opencv 彩色照片list
# output string list 识别的文字
def crnn_batch(imglist):
    alphabet = keys_crnn.alphabet
    # print(len(alphabet))
    # input('\ninput:')
    converter = util.strLabelConverter(alphabet)
    # model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
    path = './crnn/samples/model_acc97.pth'
    model.load_state_dict(torch.load(path))
    strlist = []
    # print(model)
    for i in imglist:
        img = Image.fromarray(np.array(i))
        image = img.convert('L')
        # print(image.size)
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        # print("width:" + str(w))
        transformer = dataset.resizeNormalize((w, 32))
        # image = transformer(image).cuda()
        image = transformer(image)
        image = image.view(1, *image.size())
        image = Variable(image)
        model.eval()
        preds = model(image)
        # print(preds.shape)
        _, preds = preds.max(2)
        # print(preds.shape)
        preds = preds.squeeze(1)
        preds = preds.transpose(-1, 0).contiguous().view(-1)
        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        # print(sim_pred)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))
        strlist.append(sim_pred)
    return strlist


# input: opencv 彩色照片
# output:string 识别的文字
def crnn_single(img):
    alphabet = keys_crnn.alphabet
    # print(len(alphabet))
    # input('\ninput:')
    converter = util.strLabelConverter(alphabet)
    # model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
    model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
    path = './crnn/samples/model_acc97.pth'
    model.load_state_dict(torch.load(path))
    # print(model)

    img = Image.fromarray(np.array(img))
    image = img.convert('L')
    # print(image.size)
    scale = image.size[1] * 1.0 / 32
    w = image.size[0] / scale
    w = int(w)
    # print("width:" + str(w))

    transformer = dataset.resizeNormalize((w, 32))
    # image = transformer(image).cuda()
    image = transformer(image)
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)
    # print(preds.shape)
    _, preds = preds.max(2)
    # print(preds.shape)

    # preds = preds.squeeze(2)
    # preds = preds.transpose(1, 0).contiguous().view(-1)
    preds = preds.squeeze(1)
    preds = preds.transpose(-1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return sim_pred


def process_boxes(img, boxes, scale):
    strlist = []
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue
        if box[8] >= 0.9:
            color = (0, 255, 0)
        elif box[8] >= 0.8:
            color = (255, 0, 0)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        min_x = min(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        min_y = min(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        max_x = max(int(box[0] / scale), int(box[2] / scale), int(box[4] / scale), int(box[6] / scale))
        max_y = max(int(box[1] / scale), int(box[3] / scale), int(box[5] / scale), int(box[7] / scale))
        line = str(min_x) + "," + str(min_y) + "," + str(max_x) + "," + str(max_y)
        strlist.append(line)
    return strlist, img


# input: opencv 彩色照片
# output：string list 检测区域的四个坐标
def ctpn_single(img):
    cfg_from_file('./ctpn/text.yml')
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('./ctpn/data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    blobs, im_scales = _get_blobs(img, None)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
    rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

    scores = rois[:, 0]
    boxes = rois[:, 1:5] / im_scales[0]
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    strlist, img = process_boxes(img, boxes, scale)
    cv2.imshow("detection", img)
    while (1):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyWindow("detection")
    return strlist


# input opencv list 彩色照片 string list 照片名字list
# output string 二维list 检测区域的四个坐标,opencv 框出区域的照片
def ctpn_batch(imglist):
    cfg_from_file('./ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    with gfile.FastGFile('./ctpn/data/ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')
    stroutput = []
    imgoutput = []
    for i in range(len(imglist)):
        img = imglist[i]
        #name = imgnames[i]
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)

        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        strlist, img = process_boxes(img, boxes, scale)
        stroutput.append(strlist)
        imgoutput.append(img)
        # cv2.imshow("detection", img)
        # while (1):
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        # cv2.destroyWindow("detection")
        # print(str(len(strlist)) + "个框")
        # print(strlist)
    return stroutput, imgoutput


# input opencv彩色照片，string list四个坐标的字符串
# output opencv list 照片list
def pt2img(strlist, img):
    # cv2.imshow("img", img)
    imglist = []
    for line in strlist:
        # print(line)
        pts = line.split(",")
        img_crop = img[int(pts[1]):int(pts[3]), int(pts[0]):int(pts[2])]
        # cv2.imshow(line, img_crop)
        imglist.append(img_crop)
    # while (1):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyAllWindows()
    return imglist


def str2txt(strlist, txtpath):
    f = open(txtpath, "w")
    for line in strlist:
        line = line + "\n"
        f.write(line)
    f.close()

def deletedot(strlist):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
    stroutput=[]
    for line in strlist:
        line=cop.sub("",line)
        stroutput.append(line)
    return stroutput

#input opencv 单张彩色照片
#output string list识别的文字list
def ctpn_crnn_single(img):
    strlist = ctpn_single(img)
    imglist = pt2img(strlist, img)
    result = crnn_batch(imglist)
    return result

def ctpn_crnn_batch(imglist,imgnames,savepath):
    if os.path.exists(savepath):
        shutil.rmtree(savepath)
    os.makedirs(savepath)
    str2list, imglist_detect = ctpn_batch(imglist)
    str2output=[]
    # 对每张照片进行文字识别
    for i in range(0, len(str2list)):
        imglist_tmp = pt2img(str2list[i], imglist[i])
        result_list = deletedot(crnn_batch(imglist_tmp))
        # 写每张照片识别出的文字
        str2txt(result_list, savepath + imgnames[i] + ".txt")
        # 保存检测出文本框的照片
        cv2.imwrite(os.path.join(savepath, imgnames[i] + ".jpg"), imglist_detect[i])
        #print(result_list)
        str2output.append(result_list)
    return str2output

#input string list 模板文字list，string list学生文字list，阈值
#output list 该题的正误，1是正确，0是错误
def score_one_quest_one_std(tpllist,stdlist,threshold):
    scores=[]
    for i in range(len(tpllist)):
        score=0
        for std in stdlist:
            if(Levenshtein.ratio(tpllist[i], std)>=threshold):
                score=1
        scores.append(score)
    return scores





##########单张测试###############
# im_path = "./data/demo/test1.png"
# img = cv2.imread(im_path)
# result=ctpn_crnn_single(img)
# print(result)

############多张测试################
# time_start = time.time()
# dirpath = "C:\\Users\\84019\\Desktop\\ocr_all\\ocr_all\\1-1\\"
# savepath = dirpath + "result\\"
# if os.path.exists(savepath):
#     shutil.rmtree(savepath)
# os.makedirs(savepath)
# paths = glob.glob(os.path.join(dirpath, '*.[jp][pn]g'))
# imglist = []
# imgnames = []
# for i in paths:
#     name = i.split("\\")[-1]
#     name = name.split(".")[0]
#     imgnames.append(name)
#     img = cv2.imread(i)
#     imglist.append(img)
#
# resultlist=ctpn_crnn_batch(imglist,imgnames,savepath)
#
# time_end = time.time()
# print("time cost:" + str(time_end - time_start) + "s")

############评分测试#####################
# tpllist=["nihao","beijing","shanghai"]
# stdlist=["nhao","xinjiang","bjing","chnia","shanghai","americna"]
# threshold=0.9
# print(score_one_quest_one_std(tpllist,stdlist,threshold))

##################整体测试################
time_start=time.time()
tpl_img_list=[]
tpl_names_list=[]
dirpath="C:\\Users\\84019\\Desktop\\ocr_all\\ocr_all\\1\\"
tpl_name1="answer1.png"
tpl_name2="answer2.png"
tpl_img_list.append(cv2.imread(dirpath+tpl_name1))
tpl_img_list.append(cv2.imread(dirpath+tpl_name2))
tpl_str_list=crnn_batch(tpl_img_list)
print("answer are:"+str(tpl_str_list))

std_img_list=[]
std_names_list=[]
std_name1="1-1-1-1.jpg"
std_name2="1-1-1-2.jpg"
std_name3="1-1-1-3.jpg"
std_name4="1-1-1-4.jpg"
std_name5="1-1-1-5.jpg"
std_name6="1-1-1-6.jpg"
std_name7="1-1-1-7.jpg"
std_name8="1-1-1-8.jpg"
std_name9="1-1-1-9.jpg"
std_name10="1-1-1-10.jpg"
std_name11="1-1-1-11.jpg"
std_name12="1-1-1-12.jpg"
std_name13="1-1-1-13.jpg"
std_names_list.append(std_name1.split(".")[0])
std_names_list.append(std_name2.split(".")[0])
std_names_list.append(std_name3.split(".")[0])
std_names_list.append(std_name4.split(".")[0])
std_names_list.append(std_name5.split(".")[0])
std_names_list.append(std_name6.split(".")[0])
std_names_list.append(std_name7.split(".")[0])
std_names_list.append(std_name8.split(".")[0])
std_names_list.append(std_name9.split(".")[0])
std_names_list.append(std_name10.split(".")[0])
std_names_list.append(std_name11.split(".")[0])
std_names_list.append(std_name12.split(".")[0])
std_names_list.append(std_name13.split(".")[0])
std_img_list.append(cv2.imread(dirpath+std_name1))
std_img_list.append(cv2.imread(dirpath+std_name2))
std_img_list.append(cv2.imread(dirpath+std_name3))
std_img_list.append(cv2.imread(dirpath+std_name4))
std_img_list.append(cv2.imread(dirpath+std_name5))
std_img_list.append(cv2.imread(dirpath+std_name6))
std_img_list.append(cv2.imread(dirpath+std_name7))
std_img_list.append(cv2.imread(dirpath+std_name8))
std_img_list.append(cv2.imread(dirpath+std_name9))
std_img_list.append(cv2.imread(dirpath+std_name10))
std_img_list.append(cv2.imread(dirpath+std_name11))
std_img_list.append(cv2.imread(dirpath+std_name12))
std_img_list.append(cv2.imread(dirpath+std_name13))
str2list=ctpn_crnn_batch(std_img_list,std_names_list,dirpath+"result\\")

stds_results=[]
for strlist in str2list:
    stds_results.append(score_one_quest_one_std(tpl_str_list,strlist,0.8))
print(stds_results)
time_end=time.time()
print("time_cost:"+str(time_end-time_start)+"s")
