# coding:utf-8

import dataset
import keys_crnn
import models.crnn as crnn
import torch.utils.data
import util
from PIL import Image
from torch.autograd import Variable
import glob
import os
import sys

path=os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
print(path)
sys.path.append(path)

def batch_test(dirpath):
	alphabet = keys_crnn.alphabet
	#print(len(alphabet))
	#input('\ninput:')
	converter = util.strLabelConverter(alphabet)
	# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
	model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
	path = './samples/model_acc97.pth'
	model.load_state_dict(torch.load(path))
	#print(model)
	paths=glob.glob(os.path.join(dirpath,'*.[jp][pn]g'))
	for i in paths:
		print(i)
		image = Image.open(i).convert('L')
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
		print(sim_pred)
		#print('%-20s => %-20s' % (raw_pred, sim_pred))


def single_test(imgpath):
	alphabet = keys_crnn.alphabet
	#print(len(alphabet))
	#input('\ninput:')
	converter = util.strLabelConverter(alphabet)
	# model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1).cuda()
	model = crnn.CRNN(32, 1, len(alphabet) + 1, 256, 1)
	path = './samples/model_acc97.pth'
	model.load_state_dict(torch.load(path))
	#print(model)

	im_path = imgpath
	image = Image.open(im_path).convert('L')
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
	print('%-20s => %-20s' % (raw_pred, sim_pred))

dirpath="D:/Ocr/dataset/20190316-3/BACKUP_1/CentOS-S1/crop/"
im_path = "../img/test1.png"
#single_test(im_path)
batch_test(dirpath)


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
