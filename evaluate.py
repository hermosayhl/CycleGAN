# Python
import math
# 3rd party
import cv2
import numpy
# tensorflow
import torch
import torch.nn.functional as F
from torch.autograd import Variable


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
		self.compute_ssim = SSIM()

	def update(self, label_image, pred_image):
		# 计数 + 1
		self.count += 1
		# mse loss
		mse_loss_value = self.mse_loss_fn(label_image, pred_image)
		self.mean_mse_loss += mse_loss_value.item()
		psnr_value = self.compute_psnr(mse_loss_value)
		self.mean_psnr += psnr_value
		if(not self.psnr_only):
			ssim_value = self.compute_ssim(label_image, pred_image)
			self.mean_ssim += ssim_value
		# 计算损失
		total_loss_value = 1.0 * mse_loss_value
		
		self.mean_loss += total_loss_value.item()
		return total_loss_value

	def get(self):
		if(self.count == 0):
			return 0
		if(self.psnr_only):
			return self.mean_loss / self.count, self.mean_mse_loss * (255 ** 2) / self.count, self.mean_psnr / self.count
		else:
			return self.mean_loss / self.count, self.mean_mse_loss * (255 ** 2) / self.count, self.mean_psnr / self.count, self.mean_ssim / self.count

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






def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size / 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    _padding = (int(window_size / 2), int(window_size / 2))
    mu1 = F.conv2d(img1, window, padding=_padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=_padding, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=_padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=_padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=_padding, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel).cuda()

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel:
            window = self.window
        else:
            window = create_window(self.window_size, channel).cuda()
            self.window = window
            self.channel = channel
        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)