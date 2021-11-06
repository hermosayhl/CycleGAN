# Python
import os
import cv2
import random
# 3rd party
import numpy
import torch
import dill as pickle
from torch.utils.data import Dataset



def cv2_rotate(image, angle=15):
	height, width = image.shape[:2]    
	center = (width / 2, height / 2)   
	scale = 1                        
	M = cv2.getRotationMatrix2D(center, angle, scale)
	image_rotation = cv2.warpAffine(src=image, M=M, dsize=(width, height), borderValue=(0, 0, 0))
	return image_rotation

def cv2_show(image):
	cv2.imshow('crane', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



class ImageDataset(Dataset):
	# numpy->torch.Tensor
	transform = lambda x: torch.from_numpy(x).permute(2, 0, 1).type(torch.FloatTensor).div(255)
	# torch.Tensor->numpy
	restore = lambda x: torch.clamp(x.detach().permute(0, 2, 3, 1), 0, 1).cpu().mul(255).numpy().astype('uint8')

	def __init__(self, images_list):
		self.images_list = images_list

	def __len__(self):
		return len(self.images_list)

	def __getitem__(self, idx):
		return transform(cv2.imread(self.images_list[idx]))

	@staticmethod
	def make_paired_augment(low_quality, high_quality):
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


	@staticmethod
	def prepare_paired_images(input_path, label_path, augment, target_size):
		# 读取图像
		low_quality = cv2.imread(input_path)
		high_quality = cv2.imread(label_path)
		# 数据增强
		if(augment): 
			low_quality, high_quality = ImageDataset.make_paired_augment(low_quality, high_quality)
		# 分辨率要求
		if(target_size is not None): 
			low_quality = cv2.resize(low_quality, target_size)
			high_quality = cv2.resize(high_quality, target_size)
		# numpy->tensor
		return ImageDataset.transform(low_quality), ImageDataset.transform(high_quality), os.path.split(input_path)[-1]



class PairedImageDataset(ImageDataset):
	def __init__(self, images_list, augment=True, target_size=(256, 256)):
		super(PairedImageDataset, self).__init__(images_list)
		self.augment = augment
		self.target_size = target_size

	def __getitem__(self, idx):
		# 获取路径
		input_path, label_path = self.images_list[idx]
		return ImageDataset.prepare_paired_images(input_path, label_path, self.augment, self.target_size)
		

class UnpairedImageDataset(ImageDataset):
	def __init__(self, A_images_list, B_images_list, augment=True, target_size=(256, 256)):
		super(UnpairedImageDataset, self).__init__(A_images_list)
		self.A_images_list = A_images_list
		self.B_images_list = B_images_list
		self.augment = augment
		self.target_size = target_size

	def __getitem__(self, idx):
		# 获取图像路径
		A_image_path = random.sample(self.A_images_list, k=1)[0]
		B_image_path = random.sample(self.B_images_list, k=1)[0]
		# 这里有问题, unpaired 的话, 数据增强没必要是一样的, 不过差别也不大, 后面有时间看看
		return ImageDataset.prepare_paired_images(A_image_path, B_image_path, self.augment, self.target_size)



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





if __name__ == '__main__':

	# 在这里验证下数据集是否正确?
	opt = lambda: None
	opt.dataset_dir = "D:/data/datasets/MIT-Adobe_FiveK/png"
	opt.A_dir = "input"
	opt.B_dir = "expertC_gt"
	opt.data_split = "./datasets/fiveK_split_new_unpaired.pkl"

	train_images_list, valid_images_list, test_images_list = get_unpaired_training_data(opt)
	A_images_list, B_images_list = train_images_list
	print('A  :  {}\nB  :  {}\nvalid  :  {}\ntest  :  {}'.format(len(A_images_list), len(B_images_list), len(valid_images_list), len(test_images_list)))
	
	from torch.utils.data import DataLoader

	# pair === valid
	valid_dataset = PairedImageDataset(valid_images_list, augment=False, target_size=None)
	valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=1)
	for valid_batch, (A_image, B_image, image_name) in enumerate(valid_dataloader, 1):
		A_image_back = ImageDataset.restore(A_image)[0]
		B_image_back = ImageDataset.restore(B_image)[0]
		cv2_show(numpy.concatenate([A_image_back, B_image_back], axis=1))
		if(valid_batch == 5): break

	# pair === test
	test_dataset = PairedImageDataset(test_images_list, augment=False, target_size=None)
	test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=1)
	for test_batch, (A_image, B_image, image_name) in enumerate(test_dataloader, 1):
		A_image_back = ImageDataset.restore(A_image)[0]
		B_image_back = ImageDataset.restore(B_image)[0]
		cv2_show(numpy.concatenate([A_image_back, B_image_back], axis=1))
		if(test_batch == 5): break

	# unpair === train
	train_dataset = UnpairedImageDataset(A_images_list, B_images_list, augment=False, target_size=(256, 256))
	train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1)
	for train_batch, (A_image, B_image, image_name) in enumerate(train_dataloader, 1):
		A_image_back = ImageDataset.restore(A_image)[0]
		B_image_back = ImageDataset.restore(B_image)[0]
		cv2_show(numpy.concatenate([A_image_back, B_image_back], axis=1))
		if(train_batch == 20): break


	# inference
	# ImageDataset