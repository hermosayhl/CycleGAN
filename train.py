# Python
import os
import sys
import warnings
import itertools
warnings.filterwarnings('ignore')
# 3rd party
import cv2
import numpy
import dill as pickle
# torch
import torch
from torch.utils.data import DataLoader
# self
import utils
import evaluate
import pipeline
import architectures


# ------------------------------- 定义超参等 --------------------------------------

# 参数
opt = lambda: None
# 训练
opt.gpu_id = "1"
opt.use_cuda = int(opt.gpu_id) >= 0 and torch.cuda.is_available()
opt.lr = 1e-4
opt.optimizer = torch.optim.Adam
opt.low_size = (256, 256)
opt.total_epochs = 100
opt.train_batch_size = 1
opt.valid_batch_size = 1
opt.test_batch_size = 1
# 数据
opt.dataset_name = 'fiveK'
opt.dataset_dir = "/home/dongxuan/datasets/MIT-Adobe_FiveK/"
opt.A_dir = "input"
opt.B_dir = "expertC_gt"
opt.data_split = "./datasets/fiveK_split_new_unpaired.pkl"
# 实验
opt.exp_name = "baseline"
opt.seed = 1998
opt.save = True
opt.valid_interval = 1
opt.checkpoints_dir = os.path.join("./checkpoints/", opt.exp_name)
# 可视化参数
opt.visualize_size = 1
opt.visualize_batch = 500
opt.visualize_dir = os.path.join(opt.checkpoints_dir, 'train_phase') 
# 测试参数, 测试直接把结果写出来
opt.test_save_dir = "./generated/{}".format(opt.exp_name)
for l, r in vars(opt).items(): print(l, " : ", r)
os.makedirs(opt.checkpoints_dir, exist_ok=True)
os.makedirs(opt.visualize_dir, exist_ok=True)
os.makedirs(opt.test_save_dir, exist_ok=True)
assert os.path.exists(opt.dataset_dir), "dataset for low/high quality image pairs doesn't exist !"


# 设置随机种子
utils.set_seed(seed=opt.seed, gpu_id=opt.gpu_id)


# ------------------------------- 定义数据读取 --------------------------------------

train_images_list, valid_images_list, test_images_list = pipeline.get_unpaired_training_data(opt)
A_images_list, B_images_list = train_images_list
print('A  :  {}\nB  :  {}\nvalid  :  {}\ntest  :  {}'.format(len(A_images_list), len(B_images_list), len(valid_images_list), len(test_images_list)))
# valid
valid_dataset = pipeline.PairedImageDataset(valid_images_list, augment=False, target_size=None)
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=opt.valid_batch_size)
# test
test_dataset = pipeline.PairedImageDataset(test_images_list, augment=False, target_size=None)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=opt.test_batch_size)
# train
train_dataset = pipeline.UnpairedImageDataset(A_images_list, B_images_list, augment=True, target_size=opt.low_size)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=opt.train_batch_size, worker_init_fn=utils.worker_init_fn)



# ------------------------------- 定义网络结构 --------------------------------------
network_G = architectures.Generator()
network_F = architectures.Generator()
network_D_A = architectures.Discriminator(opt.low_size)
network_D_B = architectures.Discriminator(opt.low_size)

# 送到 GPU
if(opt.use_cuda):
	network_G, network_F = network_G.cuda(), network_F.cuda()
	network_D_A, network_D_B = network_D_A.cuda(), network_D_B.cuda()

# 定义损失函数
loss_fn_G = torch.nn.MSELoss()
loss_fn_cycle = torch.nn.L1Loss()
loss_fn_D = lambda l, r: torch.mean((l - r) ** 2)

# 优化器
optimizer_G = torch.optim.Adam(itertools.chain(network_G.parameters(), network_F.parameters()), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(network_D_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(network_D_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

# 保存当前最好的一次训练结果
max_psnr = -1e3
max_psnr_checkpoint = ""


# ------------------------------- 开始训练 --------------------------------------
for epoch in range(1, opt.total_epochs + 1):
	print()
	# 计时
	with utils.Timer() as time_scope:
		# 设置 train 模式
		network_G.train()
		network_F.train()
		network_D_A.train()
		network_D_B.train()
		# 记录一些损失
		mean_loss_D_max = 0
		mean_loss_cycle = 0
		mean_loss_D_A_min = 0
		mean_loss_D_B_min = 0
		# 读取数据训练
		for train_batch, (A_image, B_image, image_name) in enumerate(train_dataloader, 1):
			# 清空生成器 G 和 F 的梯度
			optimizer_G.zero_grad()
			# 把图片放到 GPU
			if(opt.use_cuda): A_image, B_image = A_image.cuda(), B_image.cuda()
			# A -> B
			fake_B = network_G(A_image)
			# A -> B -> A
			fake_cycle_A = network_F(fake_B)
			# B -> A
			fake_A = network_F(B_image)
			# B -> A -> B
			fake_cycle_B = network_G(fake_A)
			# 对 fake_B 和 fake_A 判别真假
			fake_B_guess = network_D_B(fake_B)
			fake_A_guess = network_D_A(fake_A)
			# 计算生成器的损失(得到的结果越像目标 domain 的图像)
			loss_D_max = loss_fn_D(fake_B, 1.0) + loss_fn_D(fake_A, 1.0)
			# 计算循环一致性损失
			loss_cycle = loss_fn_cycle(fake_cycle_A, A_image) + loss_fn_cycle(fake_cycle_B, B_image)
			# 优化生成器时的总损失
			total_loss = 1.0 * loss_D_max + 10.0 * loss_cycle
			total_loss.backward(retain_graph=True)
			# w -= lr * gradient
			optimizer_G.step()
			# 这里可以删除一些东西, optimizer ?

			optimizer_D_A.zero_grad()
			optimizer_D_B.zero_grad()
			# 优化判别器, 真实图像全部打成 1, 造的图像打成 0
			A_image_guess = network_D_A(A_image)
			fake_A_guess = network_D_A(fake_A.detach())
			B_image_guess = network_D_B(B_image)
			fake_B_guess = network_D_B(fake_B.detach())
			# 计算两个判别器的损失并更新
			loss_D_A = loss_fn_D(A_image_guess, 1) + loss_fn_D(fake_A_guess, 0)
			loss_D_A.backward()
			optimizer_D_A.step()
			loss_D_B = loss_fn_D(B_image_guess, 1) + loss_fn_D(fake_B_guess, 0)
			loss_D_B.backward()
			optimizer_D_B.step()

			# 统计一些变量
			mean_loss_D_max += loss_D_max.item()
			mean_loss_cycle += loss_cycle.item()
			mean_loss_D_A_min += loss_D_A.item()
			mean_loss_D_B_min += loss_D_B.item()

			# 输出信息
			sys.stdout.write('\rTrain===>[Epoch {}/{}] [Batch {}/{}] [D_max {:.3f} - cycle {:.3f} - D_A_min {:.3f} - D_B_min {:.3f}]'.format(
				epoch, opt.total_epochs, train_batch, len(train_dataloader), \
				mean_loss_D_max / train_batch, mean_loss_cycle / train_batch, mean_loss_D_A_min / train_batch, mean_loss_D_B_min / train_batch))

			# 中途可视化一些结果
			if((train_batch - 1) % opt.visualize_batch == 0):
				# A_image, fake_B, fake_cycle_A
				# B_image, fake_A, fake_cycle_B
				R = pipeline.ImageDataset.restore
				detail_image_A = numpy.concatenate([R(A_image)[0], R(fake_B)[0], R(fake_cycle_A)[0]], axis=1)
				detail_image_B = numpy.concatenate([R(B_image)[0], R(fake_A)[0], R(fake_cycle_B)[0]], axis=1)
				detail_image = numpy.concatenate([detail_image_A, detail_image_B], axis=0)
				cv2.imwrite(os.path.join(opt.visualize_dir, "epoch_{}_batch_{}.png".format(epoch, train_batch)), detail_image)

	# 开始验证一波
	if(epoch % opt.valid_interval == 0):
		# 设置 eval 模式
		network_G.eval()
		network_F.eval()
		network_D_A.eval()
		network_D_B.eval()
		# 开始遍历验证集, 这里只算 psnr ? 简易版本的, 命名为一个 train_simple.py, 后面再慢慢完善
		valid_evaluator = evaluate.PairedImageEvaluator()
		with torch.no_grad():
			for valid_batch, (A_image, B_image, image_name) in enumerate(valid_dataloader, 1):
				if(opt.use_cuda): A_image, B_image = A_image.cuda(), B_image.cuda()
				fake_B = network_G(A_image)
				valid_evaluator.update(fake_B, B_image)
				sys.stdout.write('\rValid===>[Epoch {}/{}] [Batch {}/{}] [MSE {:.1f}] [PSNR {:.3f}] - [SSIM {:.3f}]'.format(
					epoch, opt.total_epochs, valid_batch, len(valid_dataloader), *valid_evaluator.get()))
			# 检查性能, 然后保存模型
			_, valid_psnr, valid_ssim = valid_evaluator.get()
			if(valid_psnr > max_psnr):
				max_psnr = valid_psnr
				save_name = "epoch_{}_psnr_{:.3f}_ssim_{:.3f}.pth".format(epoch, valid_psnr, valid_ssim)
				max_psnr_checkpoint = os.path.join(opt.checkpoints_dir, save_name)
				torch.save({"network_G": network_G.state_dict()}, max_psnr_checkpoint)
				print('\nsaved to ===> {}\n'.format(max_psnr_checkpoint))

# ------------------------------- 开始测试 --------------------------------------
del network_F
del network_D_A
del network_D_B

print("\n开始处理测试集......")

# 首先加载之前最好的一次权重
state_dict = torch.load(max_psnr_checkpoint)
network_G.load_state_dict(state_dict['network_G'])

# 设置 eval 模式
network_G.eval()

test_evaluator = evaluate.PairedImageEvaluator()
with torch.no_grad():
	R = pipeline.ImageDataset.restore
	for test_batch, (A_image, B_image, image_name) in enumerate(test_dataloader, 1):
		if(opt.use_cuda): A_image, B_image = A_image.cuda(), B_image.cuda()
		fake_B = network_G(A_image)
		test_evaluator.update(fake_B, B_image)
		# 保存图像
		cv2.imwrite(os.path.join(opt.test_save_dir, image_name[0]), R(fake_B)[0])
		sys.stdout.write('\rTest===>[Epoch {}/{}] [Batch {}/{}] [MSE {:.1f}] [PSNR {:.3f}] - [SSIM {:.3f}]'.format(
			epoch, opt.total_epochs, test_batch, len(test_dataloader), *test_evaluator.get()))