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
from fuzzywuzzy import fuzz
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
        #print("cropimg size"+str(i.shape))
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
        sim_pred=sim_pred.lower()
        # print(sim_pred)
        # print('%-20s => %-20s' % (raw_pred, sim_pred))
        strlist.append(sim_pred)
    return deletedot(strlist)

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
    sim_pred=sim_pred.lower()
    # print('%-20s => %-20s' % (raw_pred, sim_pred))
    return deletedot(sim_pred)


def process_boxes(img, boxes, scale):
    listoutput = []
    #print("img size"+str(img.shape))
    h=round(img.shape[0]/scale)
    w=round(img.shape[1]/scale)
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
        #print("更改前："+str(min_x) + "\t" +str(min_y) + "\t" + str(max_x) + "\t" +str(max_y))
        if min_y<5:
            min_y=0
        else:
            min_y=min_y-5
        if (max_x+5)>w:
            max_x=w
        else:
            max_x=max_x+5
        if (max_y+2)>h:
            max_y=h
        else:
            max_y=max_y+2
        linestring = str(min_x) + "\t" +str(min_y) + "\t" + str(max_x) + "\t" +str(max_y)
        #print("更改后："+linestring)
        listoutput.append(linestring)
    img = cv2.resize(img, None, None, fx=1.0 / scale, fy=1.0 / scale, interpolation=cv2.INTER_LINEAR)
    return listoutput, img


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
    #print(img.shape)
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
    #print(boxes)
    strlist, img = process_boxes(img, boxes, scale)
    # cv2.imshow("detection", img)
    # while (1):
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cv2.destroyWindow("detection")
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
        pts = line.split("\t")
        img_crop = img[int(pts[1]):int(pts[3]), int(pts[0]):int(pts[2])]
        #print(img_crop.shape)
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
            if (fuzz.partial_ratio(tpllist[i], std) >= threshold * 100):
            #if(Levenshtein.ratio(tpllist[i], std)>=threshold):
                score=1
        scores.append(score)
    return scores

################match.py###################

######命令行传值###############
# strinput=sys.argv[1].split(',')
# tpl=[]
# for i in range(len(strinput)):
#     tpl.append(cv2.imdecode(np.fromfile(strinput[i],dtype=np.uint8),1))
#     #tpl_tmp=cv2.imread(strinput[i],1)
#     #print(strinput[i])
#     #print("tpl_tmp:"+str(tpl_tmp.shape))
#     #tpl.append(tpl_tmp)
# target =cv2.imdecode(np.fromfile(sys.argv[2],dtype=np.uint8),1)
# #target=cv2.imread(sys.argv[2],1)
# location_txt=sys.argv[3]
# result_txt=sys.argv[4]
###############################

######测试传值#################
# strinput=[]
# strinput.append("D:\\Ocr\\data0502\\standard\\1VLAN_1.png")
# strinput.append("D:\\Ocr\\data0502\\standard\\1VLAN_2.png")
# strinput.append("D:\\Ocr\\data0502\\standard\\1VLAN_3.png")
# strinput.append("D:\\Ocr\\data0502\\standard\\1VLAN_4.png")
# tpl=[]
# for i in range(len(strinput)):
#     tpl.append(cv2.imread(strinput[i],1))
# studentdirpath="D:\\Ocr\\data0502\\student\\1\\"
# if os.path.exists(studentdirpath+"result\\"):
#     shutil.rmtree(studentdirpath+"result\\")
# os.makedirs(studentdirpath+"result\\")
# target =cv2.imread(studentdirpath+"1VLAN.PNG",1)
# location_txt=studentdirpath+"result\\1_location.txt"
# result_txt=studentdirpath+"result\\1_result.txt"
###############################

# tcd=target
# #print(len(tpl))
# file_location=open(location_txt,"w")
# file_res = open(result_txt, 'w')
#
# tpl_str_list=crnn_batch(tpl)
# print("answers are:"+str(tpl_str_list))
#
# location_list=ctpn_single(target)
# for location in location_list:
#     file_location.write(location+"\n")
# file_location.close()
# img_temp_list=pt2img(location_list,target)
# std_str_list=crnn_batch(img_temp_list)
# print("学生识别："+str(std_str_list))
# scores=score_one_quest_one_std(tpl_str_list,std_str_list,0.7)
# for score in scores:
#     file_res.write(str(score)+"\n")
# file_res.close()




##########单张测试###############
#修改im_path即可，在命令行输入 python for_internet.py即可输出list string
im_path = "D:\\Ocr\\data0502\\student\\1\\1VLAN.PNG"

img = cv2.imread(im_path)
print(img.shape)
result=ctpn_crnn_single(img)
print(result)

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
# time_start=time.time()
# tpl_img_list=[]
# tpl_names_list=[]
# tpldirpath="D:\\Ocr\\江苏赛题\\标准答案截图\\"
# tpl_name1="1VLAN_1.png"
# tpl_name2="1VLAN_2.png"
# tpl_name3="1VLAN_3.png"
# tpl_name4="1VLAN_4.png"
# tpl_img_list.append(cv2.imread(tpldirpath+tpl_name1))
# tpl_img_list.append(cv2.imread(tpldirpath+tpl_name2))
# tpl_img_list.append(cv2.imread(tpldirpath+tpl_name3))
# tpl_img_list.append(cv2.imread(tpldirpath+tpl_name4))
# tpl_str_list=crnn_batch(tpl_img_list)
# print("answer are:"+str(tpl_str_list))
#
# stddirpath="D:\\Ocr\\江苏赛题\\选手答案截图\\第一组\\"
# std_img_list=[]
# std_names_list=[]
# std_name1="1VLAN.PNG"
#
# std_names_list.append(std_name1.split(".")[0])
#
# std_img_list.append(cv2.imread(dirpath+std_name1))
#
# str2list=ctpn_crnn_batch(std_img_list,std_names_list,dirpath+"result\\")
#
# stds_results=[]
# for strlist in str2list:
#     stds_results.append(score_one_quest_one_std(tpl_str_list,strlist,0.8))
# print(stds_results)
# time_end=time.time()
# print("time_cost:"+str(time_end-time_start)+"s")

