# Python
import math
# 3rd party
import cv2
import numpy
# tensorflow
import torch


###########################################################################
#                                Metrics
###########################################################################


class ImageEnhanceEvaluator():

	def __init__(self, psnr_only=True):
		self.psnr_only = psnr_only
		# mse 损失函数
		self.mse_loss_fn = torch.nn.MSELoss()
		# 统计一些值
		self.mean_psnr = 0
		self.mean_ssim = 0
		self.mean_loss = 0
		self.mean_mse_loss = 0
		# 统计第几次
		self.count = 0
		# 根据 mse_loss 计算 psnr
		self.compute_psnr = lambda mse: 10 * torch.log10(1. / mse).item() if(mse > 1e-5) else 50

	def update(self, label_image, pred_image):
		# 计数 + 1
		self.count += 1
		# mse loss
		mse_loss_value = self.mse_loss_fn(label_image, pred_image)
		self.mean_mse_loss += mse_loss_value.item()

		psnr_value = self.compute_psnr(mse_loss_value)
		self.mean_psnr += psnr_value
		# 计算损失
		total_loss_value = 1.0 * mse_loss_value
		
		self.mean_loss += total_loss_value.item()
		return total_loss_value

	def get(self):
		if(self.count == 0):
			return 0
		if(self.psnr_only):
			return self.mean_loss / self.count, self.mean_mse_loss * (255 ** 2) / self.count, self.mean_psnr / self.count

	def clear(self):
		self.count = 0
		self.mean_psnr = self.mean_ssim = self.mean_mse_loss = self.mean_loss = self.mean_tv_loss = self.mean_color_loss = 0





class LossLogger():
	def __init__(self, ):
		self.mean_loss_G_A2B = 0
		self.mean_loss_G_B2A = 0
		self.mean_loss_D_B = 0
		self.mean_loss_D_A = 0
		self.mean_loss_cycle_A2B2A = 0
		self.mean_loss_cycle_B2A2B = 0
		self.count = 0

	def update(self, loss_G_A2B, loss_G_B2A, loss_D_B, loss_D_A, loss_cycle_A2B2A, loss_cycle_B2A2B):
		self.count += 1
		self.mean_loss_G_A2B += loss_G_A2B
		self.mean_loss_G_B2A += loss_G_B2A
		self.mean_loss_D_B += loss_D_B
		self.mean_loss_D_A += loss_D_A
		self.mean_loss_cycle_A2B2A += loss_cycle_A2B2A
		self.mean_loss_cycle_B2A2B += loss_cycle_B2A2B

	def get(self):
		if(self.count == 0):
			return 0, 0, 0, 0, 0, 0
		return  self.mean_loss_G_A2B / self.count, \
				self.mean_loss_G_B2A / self.count, \
				self.mean_loss_D_B / self.count, \
				self.mean_loss_D_A / self.count, \
				self.mean_loss_cycle_A2B2A / self.count, \
				self.mean_loss_cycle_B2A2B / self.count