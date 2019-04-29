# coding:utf-8

from crnn import dataset
from crnn import keys_crnn
from crnn.models import crnn
import torch.utils.data
from crnn import util
from PIL import Image
from torch.autograd import Variable
import glob
import os
import sys
import cv2
import numpy as np

path=os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
print(path)
sys.path.append(path)
sys.path.append(os.getcwd())

#input opencv 彩色照片list
#output string list 识别的文字
def batch_test(imglist):
	alphabet = keys_crnn.alphabet
	#print(len(alphabet))
	#input('\ninput:')
	converter = util.strLabelConverter(alphabet)
	# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
	model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
	path = './crnn/samples/model_acc97.pth'
	model.load_state_dict(torch.load(path))
	strlist=[]
	#print(model)
	for i in imglist:
		img=Image.fromarray(np.array(i))
		image = img.convert('L')
		#print(image.size)
		scale = image.size[1] * 1.0 / 32
		w = image.size[0] / scale
		w = int(w)
		#print("width:" + str(w))
		transformer = dataset.resizeNormalize((w, 32))
		# image = transformer(image).cuda()
		image = transformer(image)
		image = image.view(1, *image.size())
		image = Variable(image)
		model.eval()
		preds = model(image)
		#print(preds.shape)
		_, preds = preds.max(2)
		#print(preds.shape)
		preds = preds.squeeze(1)
		preds = preds.transpose(-1, 0).contiguous().view(-1)
		preds_size = Variable(torch.IntTensor([preds.size(0)]))
		raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
		sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
		#print(sim_pred)
		#print('%-20s => %-20s' % (raw_pred, sim_pred))
		strlist.append(sim_pred)
	return strlist

#input: opencv 彩色照片
#output:string 识别的文字
def single_test(img):
	alphabet = keys_crnn.alphabet
	#print(len(alphabet))
	#input('\ninput:')
	converter = util.strLabelConverter(alphabet)
	# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
	model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
	path = './crnn/samples/model_acc97.pth'
	model.load_state_dict(torch.load(path))
	#print(model)

	img=Image.fromarray(np.array(img))
	image = img.convert('L')
	#print(image.size)
	scale = image.size[1] * 1.0 / 32
	w = image.size[0] / scale
	w = int(w)
	#print("width:" + str(w))

	transformer = dataset.resizeNormalize((w, 32))
	# image = transformer(image).cuda()
	image = transformer(image)
	image = image.view(1, *image.size())
	image = Variable(image)

	model.eval()
	preds = model(image)
	#print(preds.shape)
	_, preds = preds.max(2)
	#print(preds.shape)

	# preds = preds.squeeze(2)
	# preds = preds.transpose(1, 0).contiguous().view(-1)
	preds = preds.squeeze(1)
	preds = preds.transpose(-1, 0).contiguous().view(-1)

	preds_size = Variable(torch.IntTensor([preds.size(0)]))
	raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
	sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
	#print('%-20s => %-20s' % (raw_pred, sim_pred))
	return sim_pred



##########单张测试##############
# im_path = "./data/demo/test1.png"
# img=cv2.imread(im_path)
# print(single_test(img))

#########多张测试###############
dirpath="./data/demo/"
paths=glob.glob(os.path.join(dirpath,'*.[jp][pn]g'))
imglist=[]
for i in paths:
	img=cv2.imread(i)
	imglist.append(img)
print(batch_test(imglist))


# alphabet = keys_crnn.alphabet
# print(len(alphabet))
# input('\ninput:')
# converter = util.strLabelConverter(alphabet)
# # model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
# path = './samples/model_acc97.pth'
# model.load_state_dict(torch.load(path))
# print(model)
#
# im_path = "../img/test1.png"
# image = Image.open(im_path).convert('L')
# print(image.size)
# scale = image.size[1] * 1.0 / 32
# w = image.size[0] / scale
# w = int(w)
# print("width:" + str(w))
#
# transformer = dataset.resizeNormalize((w, 32))
# # image = transformer(image).cuda()
# image = transformer(image)
# image = image.view(1, *image.size())
# image = Variable(image)
#
# model.eval()
# preds = model(image)
# print(preds.shape)
# _, preds = preds.max(2)
# print(preds.shape)
#
# #preds = preds.squeeze(2)
# #preds = preds.transpose(1, 0).contiguous().view(-1)
# preds = preds.squeeze(1)
# preds = preds.transpose(-1, 0).contiguous().view(-1)
#
# preds_size = Variable(torch.IntTensor([preds.size(0)]))
# raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
# sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
# print('%-20s => %-20s' % (raw_pred, sim_pred))


# while 1:
#     im_name = input("\nplease input file name:")
#     im_path = "../img/" + im_name
#     image = Image.open(im_path).convert('L')
#     print(image.size)
#     scale = image.size[1] * 1.0 / 32
#     w = image.size[0] / scale
#     w = int(w)
#     print("width:"+str(w))
#
#     transformer = dataset.resizeNormalize((w, 32))
#     #image = transformer(image).cuda()
#     image = transformer(image)
#     image = image.view(1, *image.size())
#     image = Variable(image)
#     model.eval()
#     preds = model(image)
#     _, preds = preds.max(2)
#     preds = preds.squeeze(2)
#     preds = preds.transpose(1, 0).contiguous().view(-1)
#     preds_size = Variable(torch.IntTensor([preds.size(0)]))
#     raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
#     sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
#     print('%-20s => %-20s' % (raw_pred, sim_pred))
