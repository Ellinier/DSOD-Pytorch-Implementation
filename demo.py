import argparse
import os

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from DSOD import DSOD
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont

from config import cfg
from visualize_det import visualize_det

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--params_from', type=int, default=220, help='params from which checkpoint')
	args = parser.parse_args()

	net = DSOD()
	checkpoint = torch.load('./checkpoint/ckpt_{:03d}.pth'.format(args.params_from))
	net.load_state_dict(checkpoint['net'])
	net.eval()

	use_cuda = torch.cuda.is_available()

	if use_cuda:
		net.cuda()
		cudnn.benchmark = True

	data_encoder = DataEncoder()

	normMean = [0.485, 0.456, 0.406]
	normStd = [0.229, 0.224, 0.225]
	normTransform = transforms.Normalize(normMean, normStd)

	self.transform = transforms.Compose([
		transforms.Scale((300, 300)),
		transforms.ToTensor(),
		normTransform
		])

	img_dir = cfg.root
	res_dir = './results/'
	# do not know
	img_names = os.listdir(img_dir)

	f_test = open(os.path.join(os.path.join(img_dir.strip().split('/')[:-1]),'ImageSets/Main/test.txt'))

	img_names = []
	for line in f_test:
		img_name = line[:-1] + '.jpg'
		# Do not know
		img_names.append(img_name)

	for fname in img_names:
		img = Image.open(img_dir+fname)
		img_tensor = transform(img)
		if use_cuda:
			img_tensor = img_tensor.cuda()
		loc, conf = net(Variable(img_tensor[None,:,:,:], volatile=True))
		if use_cuda:
			loc = loc.cpu()
			conf = conf.cpu()

		boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
		draw = ImageDraw.Draw(img)
		if boxes is None:
			print(fname)
			continue

		img = visualize_det(img, boxes, labels, scores)
		img.save(res_dir+'det_'+fname)
