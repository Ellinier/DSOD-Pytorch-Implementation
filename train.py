from __future__ import print_function

import os
import argparse
import numpy as np
import random
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from DSOD import DSOD
from datagen import ListDataset
from multibox_loss import MultiBoxLoss
from encoder import DataEncoder
from visualize_det import visualize_det
from config import cfg

import visdom
import make_graph
<<<<<<< HEAD
viz = visdom.Visdom()
use_cuda = torch.cuda.is_available()
=======
# viz = visdom.Visdom()
# use_cuda = torch.cuda.is_available()
>>>>>>> 3ec299bb8506decc223eb6d7d9579860b1a29c01

# import shutil
# import setproctitle

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--batchSz', type=int, default=1, help='batch size')
	parser.add_argument('--nEpochs', type=int, default=300, help='number of epoch to end training')
	parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
	parser.add_argument('--momentum', type=float, default=0.9)
	parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
	# parser.add_argument('--save')
	# parser.add_argument('--seed', type=int, default=1)
	parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'adam', 'rmsprop'))
	parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
	parser.add_argument('--resume_from', type=int, default=220, help='resume from which checkpoint')
	parser.add_argument('--visdom', '-v', action='store_true', help='use visdom for training visualization')
	args = parser.parse_args()

	# args.save = args.save or 'work/DSOS.base'
	# setproctitle.setproctitle(args.save)
	# if os.path.exists(args.save):
	# 	shutil.rmtree(args.save)
	# os.makedirs(args.save, exist_ok=True)

	# use_cuda = torch.cuda.is_available()
	best_loss = float('inf') # best test loss
	start_epoch = 0 # start from epoch 0 for last epoch

	normMean = [0.485, 0.456, 0.406]
	normStd = [0.229, 0.224, 0.225]
	normTransform = transforms.Normalize(normMean, normStd)

	trainTransform = transforms.Compose([
		transforms.Scale((300, 300)),
		transforms.ToTensor(),
		normTransform
		])

	testTransform = transforms.Compose([
		transforms.Scale((300, 300)),
		transforms.ToTensor(),
		normTransform
		])

	# Data
	kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
	trainset = ListDataset(root=cfg.img_root, list_file=cfg.label_train,
		                   train=True, transform=trainTransform)
	trainLoader = DataLoader(trainset, batch_size=args.batchSz,
		                     shuffle=True, **kwargs)
	testset = ListDataset(root=cfg.img_root, list_file=cfg.label_test,
		                  train=False, transform=testTransform)
	testLoader = DataLoader(testset, batch_size=args.batchSz,
		                    shuffle=False, **kwargs)
 
	# Model
	net = DSOD(growthRate=48, reduction=1)
	if args.resume:
		print('==> Resuming from checkpoint...')
		checkpoint = torch.load('./checkpoint/ckpt_{:03d}.pth'.format(args.resume_from))
		net.load_state_dict(checkpoint['net'])
		best_loss = checkpoint['loss']
		start_epoch = checkpoint['epoch']+1
		print('Previours_epoch: {}, best_loss: {}'.format(start_epoch-1, best_loss))
	else:
		print('==> Initializing weight...')
		def init_weights(m):
			if isinstance(m, nn.Conv2d):
				init.xavier_uniform(m.weight.data)
				# m.bias.data.zero_()
		net.apply(init_weights)

	print(' + Number of params: {}'.format(
		sum([p.data.nelement() for p in net.parameters()])))
	if use_cuda:
		net = net.cuda()

	if args.opt == 'sgd':
		optimizer = optim.SGD(net.parameters(), lr=args.lr,
			                  momentum=args.momentum, weight_decay=args.wd)
	elif args.opt == 'adam':
		optimizer = optim.Adam(net.parameters(), weight_decay=args.wd)
	elif args.opt == 'rmsprop':
		optimizer = optim.RMSprop(net.parameters(), weight_decay=args.wd)

	criterion = MultiBoxLoss()

	if use_cuda:
		net.cuda()
		cudnn.benchmark = True

	if args.visdom:
		# import visdom
		# viz = visdom.Visdom()
		training_plot = viz.line(
			X=torch.zeros((1,)).cpu(),
			Y=torch.zeros((1, 3)).cpu(),
			opts=dict(
				xlabel='Epoch',
				ylabel='Loss',
				title='Epoch DSOD Training Loss',
				legend=['Loc Loss', 'Conf Loss', 'Loss']
				)
			)
		testing_plot = viz.line(
			X=torch.zeros((1,)).cpu(),
			Y=torch.zeros((1, 3)).cpu(),
			opts=dict(
				xlabel='Epoch',
				ylabel='Loss',
				title='Epoch DSOD Testing Loss',
				legend=['Loc Loss', 'Conf Loss', 'Loss']
				)
			)

	with open(cfg.label_test) as f:
		test_lines = f.readlines()
		num_tests = len(test_lines)

		transform = trainTransform
		transform_viz = testTransform

		data_encoder = DataEncoder()

		testing_image = viz.image(np.ones((3, 300, 300)),
			                      opts=dict(caption='Random Testing Image'))

	# TODO: save training data on log file
	# trainF = open(os.path.join(args.save, 'train.csv'), 'w')
	# testF = open(os.path.join(args.save, 'test.csv'), 'w')

	for epoch in range(start_epoch, start_epoch+args.nEpochs+1):
		adjust_opt(args.opt, optimizer, epoch)
<<<<<<< HEAD
		train(epoch, net, trainLoader, optimizer, criterion)
=======
		train(epoch, net, trainLoader, optimizer, criterion, viz)
>>>>>>> 3ec299bb8506decc223eb6d7d9579860b1a29c01
		test(epoch, net, testLoader, optimizer)

		if epoch%10 == 0:
			state = {
			      'net': net.state_dict(),
			      'loss': test_loss,
			      'epoch': epoch
			}
			if not os.path.isdir('checkpoint'):
				os.mkdir('checkpoint')
			torch.save(state, './checkpoint/ckpt_{:03d}.pth'.format(epoch))
		# torch.save(net, os.path.join(args.save, 'latest.path'))
		# os.system('./plot.py {} &'.format(args.save))

	# trainF.close()
	# testF.close()


<<<<<<< HEAD
def train(epoch, net, trainLoader, optimizer, criterion):
=======
def train(epoch, net, trainLoader, optimizer, criterion, viz):
>>>>>>> 3ec299bb8506decc223eb6d7d9579860b1a29c01
	print('\n==> Training Epoch %4d' % epoch)
	net.train()
	train_loss = 0
	train_loss_loc = 0
	train_loss_conf = 0

	for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainLoader):
		if use_cuda:
			images = images.cuda()
			loc_targets = loc_targets.cuda()
			conf_targets = conf_targets.cuda()

		images = Variable(images)
		loc_targets = Variable(loc_targets)
		conf_targets = Variable(conf_targets)

		optimizer.zero_grad()
		loc_preds, conf_preds = net(images)
		# print(loc_targets.size())
		# print(conf_targets.size())
		loc_loss, conf_loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
		loss = loc_loss + conf_loss
		# g = make_graph.make_dot(loss)
		# g.save('/home/ellin/Downloads/graph.dot')
		# g.view()
		# make_graph.save('/home/ellin/Downloads/graph.dot', loss.creator)
		loss.backward()
		optimizer.step()

		train_loss += loss.data[0]
		train_loss_loc += loc_loss.data[0]
		train_loss_conf += conf_loss.data[0]
		print('Epoch %4d[%3d] -> loc_loss: %.3f conf_loss: %.3f ave_loss: %.3f'
		      %(epoch, batch_idx, loc_loss.data[0], conf_loss.data[0], train_loss/(batch_idx+1)))

	if args.visdom:
		viz.line(
			X=torch.ones((1, 3)).cpu() * epoch,
			Y=torch.Tensor([train_loss_loc, train_loss_conf, train_loss]).unsequeeze(0).cpu() / (batch_idx+1),
			win=training_plot,
			update='append'
			)

def test(epoch, net, testLoader, optimizer, criterion):
	print('Testing')
	net.eval()
	test_loss = 0
	test_loss_loc = 0
	test_loss_conf = 0
	for batch_index, (images, loc_targets, conf_targets) in enumerate(testLoader):
		if use_cuda:
			images = images.cuda()
			loc_targets = loc_targets.cuda()
			conf_targets = conf_targets.cuda()

		images = Variable(images, volatile=True)
		loc_targets = Variable(loc_targets)
		conf_targets = Variable(conf_targets)

		loc_preds, conf_preds = net(images)
		loc_loss, conf_loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
		loss = loc_loss + conf_loss

		test_loss += loss.data[0]
		test_loss_loc += loc_loss.data[0]
		test_loss_conf += conf_loss.data[0]
		print('loc_loss: %.3f conf_loss: %.3f ave_loss: %.3f'
			  %(loc_loss.data[0], conf_loss.data[0], test_loss/(batch_idx+1)))

	if args.visdom:
		viz.line(
			X=torch.ones((1, 3)).cpu() * epoch,
			Y=torch.Tensor([test_loss_loc, test_loss_conf, test_loss]).unsequeeze(0).cpu() / (batch_idx+1),
			win=testing_plot,
			update='append'
			)

		ii = random.randint(0, num_tests-1)
		test_name = test_lines[ii].strip().split()[0]
		test_img = Image.open(cfg.img_root+test_name)

		img_tensor = transform(test_img)

		if use_cuda:
			img_tensor = img_tensor.cuda()
		loc, conf = net(Variable(img_tensor[None,:,:,:], volatile=True))
		if use_cuda:
			loc = loc.cpu()
			conf = conf.cpu()

		boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
		test_img = visualize_det(test_img, boxes, labels, scores)

		viz.image(
			transform_viz(test_img),
			win=testing_image,
			opts=dict(caption= 'Random Testing Image')
			)


def adjust_opt(optAlg, optimizer, epoch):
	if optAlg == 'sgd':
		if epoch < 150:
			lr = 1e-1
		elif epoch == 150:
			lr = 1e-2
		elif epoch == 225:
			lr = 1e-3
		else:
			return

		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

if __name__ == '__main__':
	main()