import os
from PIL import Image
import cv2
import numpy
import torch
import random
import dill as pickle


# 展示图像
def show(image, name='yhl'):
	cv2.imshow(name, image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


# 图片旋转
def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation



def make_augment(low_quality, high_quality):
	# 以 0.6 的概率作数据增强
	if(random.random() > 1 - 0.9):
		# 待增强操作列表(如果是 Unet 的话, 其实这里可以加入一些旋转操作)
		all_states = ['crop', 'flip', 'rotate']
		# 打乱增强的顺序
		random.shuffle(all_states)
		for cur_state in all_states:
			if(cur_state == 'flip'):
				# 0.5 概率水平翻转
				if(random.random() > 0.5):
					low_quality = cv2.flip(low_quality, 1)
					high_quality = cv2.flip(high_quality, 1)
					# print('水平翻转一次')
			elif(cur_state == 'crop'):
				# 0.5 概率做裁剪
				if(random.random() > 1 - 0.8):
					H, W, _ = low_quality.shape
					ratio = random.uniform(0.75, 0.95)
					_H = int(H * ratio)
					_W = int(W * ratio)
					pos = (numpy.random.randint(0, H - _H), numpy.random.randint(0, W - _W))
					low_quality = low_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					high_quality = high_quality[pos[0]: pos[0] + _H, pos[1]: pos[1] + _W]
					# print('裁剪一次')
			elif(cur_state == 'rotate'):
				# 0.2 概率旋转
				if(random.random() > 1 - 0.1):
					angle = random.randint(-15, 15)  
					low_quality = cv2_rotate(low_quality, angle)
					high_quality = cv2_rotate(high_quality, angle)
					# print('旋转一次')
	return low_quality, high_quality




class AlignedDataset(torch.utils.data.Dataset):

	def transform(self, x):
		return torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor).div(255)

	def restore(self, x):
		return x.detach().cpu().mul(255).permute(0, 2, 3, 1).numpy().astype('uint8')

	def __init__(self, images_list, mode='valid'):
		super(AlignedDataset, self).__init__()
		
		self.images_list = images_list
		self.mode = mode

	def __len__(self):
		return len(self.images_list)

	def __getitem__(self, idx):
		# 读取图像
		input_path, label_path = self.images_list[idx]
		low_quality = cv2.imread(input_path)
		high_quality = cv2.imread(label_path)

		# 还会对验证集做数据增强
		if(self.mode != 'test'):
			low_quality, high_quality = make_augment(low_quality, high_quality)

		return {'A': self.transform(low_quality), 'B': self.transform(high_quality), 'A_paths': input_path, 'B_paths': input_path}




class UnAlignedDataset(torch.utils.data.Dataset):

	def transform(self, x):
		return torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor).div(255)

	def restore(self, x):
		return x.detach().cpu().mul(255).permute(0, 2, 3, 1).numpy().astype('uint8')

	def __init__(self, A_images_list, B_images_list, mode='train'):
		super(UnAlignedDataset, self).__init__()

		self.A_images_list = A_images_list
		self.B_images_list = B_images_list

		self.mode = mode

	def __len__(self):
		return len(self.A_images_list)

	def __getitem__(self, idx):
		# A 列表随机选一张图像
		input_path = random.sample(self.A_images_list, k=1)[0]
		label_path = random.sample(self.B_images_list, k=1)[0]

		# 读取
		low_quality = cv2.imread(input_path)
		high_quality = cv2.imread(label_path)

		# 数据增强
		low_quality, high_quality = make_augment(low_quality, high_quality)

		# train 可能需要 batch, 所以这里要归一化
		low_quality = cv2.resize(low_quality, (256, 256))
		high_quality = cv2.resize(high_quality, (256, 256))

		return {'A': self.transform(low_quality), 'B': self.transform(high_quality), 'A_paths': input_path, 'B_paths': input_path}





def get_unpaired_training_data(opt):
	# 读取数据划分
	with open(opt.data_split, 'rb') as reader:
		data_split = pickle.load(reader)
		# 处理成对的数据, test 跟 valid
		test_images_list = [(os.path.join(opt.dataset_dir, opt.A_dir, image_name), os.path.join(opt.dataset_dir, opt.B_dir, image_name)) for image_name in data_split['test']]
		valid_images_list = [(os.path.join(opt.dataset_dir, opt.A_dir, image_name), os.path.join(opt.dataset_dir, opt.B_dir, image_name)) for image_name in data_split['valid']]
		# 处理非成对的数据
		A_images_list = [os.path.join(opt.dataset_dir, opt.A_dir, image_name) for image_name in data_split['train_A']]
		B_images_list = [os.path.join(opt.dataset_dir, opt.B_dir, image_name) for image_name in data_split['train_B']]
		# 返回列表
		return (A_images_list, B_images_list), valid_images_list, test_images_list
