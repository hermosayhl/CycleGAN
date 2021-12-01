# Python
import os
import sys
import itertools
import functools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# 3rd party
import cv2
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
# self
import utils
import pipeline
import evaluate
import architectures

# 是否 use_cuda 这里还没有写好
yaml_path = sys.argv[2]
if(not os.path.exists(yaml_path)):
    yaml_path = "./options/train/train_fivek_unpaired.yaml"

# 读取配置文件
with open(yaml_path, 'r') as yaml_reader:
    config = yaml.load(yaml_reader, Loader=yaml.FullLoader)
    opt = utils.dict2object(config)
    # 更新
    opt.intermediate.checkpoints_dir = os.path.join(opt.intermediate.checkpoints_dir, opt.name)
    opt.intermediate.visualize_dir = os.path.join(opt.intermediate.checkpoints_dir, opt.intermediate.visualize_dir)
    os.makedirs(opt.intermediate.checkpoints_dir, exist_ok=True)
    os.makedirs(opt.intermediate.visualize_dir, exist_ok=True)

# 是否可复现
if(opt.common.deterministic == True):
    # 设置种子和 GPU 环境(速度会变慢)
    utils.set_seed(opt.common.seed, str(opt.common.gpu_id))
else:
    # 随便一个简单环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.common.gpu_id)
    torch.set_num_threads(4)
    torch.set_default_tensor_type(torch.FloatTensor)


# 获取数据
train_images_list, valid_images_list, test_images_list = pipeline.get_unpaired_training_data(opt)
A_images_list, B_images_list = train_images_list
# 构造数据读取流
train_dataset = pipeline.UnAlignedDataset(A_images_list, B_images_list, mode='train', low_res=opt.datasets.low_res)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=opt.train.train_batch_size, 
    shuffle=True,
    worker_init_fn=utils.worker_init_fn,
    num_workers=0)
valid_dataset = pipeline.AlignedDataset(valid_images_list, mode='valid')
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, 
    batch_size=opt.valid.valid_batch_size,
    shuffle=False,
    worker_init_fn=utils.worker_init_fn,
    # 为了泛化, 对验证数据集进行重复采样, 而且还会进行数据增强
    sampler=torch.utils.data.RandomSampler(valid_dataset, replacement=True, num_samples=opt.valid.repeat * len(valid_dataset)))

# 设定网络结构
netG_A2B = architectures.Generator()
netG_B2A = architectures.Generator()
# netD_B = architectures.NLayerDiscriminator(3, 64, n_layers=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False))
# netD_A = architectures.NLayerDiscriminator(3, 64, n_layers=3, norm_layer=functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False))
netD_B = architectures.Discriminator(low_res=opt.datasets.low_res)
netD_A = architectures.Discriminator(low_res=opt.datasets.low_res)

# 网络送到 GPU
netG_A2B = netG_A2B.cuda()
netG_B2A = netG_B2A.cuda()
netD_B = netD_B.cuda()
netD_A = netD_A.cuda()

# 默认都是 train 模式
netG_A2B.train()
netG_B2A.train()
netD_B.train()
netD_A.train()

# 之前生成的 fake 图像的历史
fake_A_pool = utils.ImagePool(opt.train.history_size)  
fake_B_pool = utils.ImagePool(opt.train.history_size) 

# 损失函数
loss_adversarial = torch.nn.MSELoss().cuda()
loss_cycle = torch.nn.L1Loss()
loss_identity = torch.nn.L1Loss()

# 为网络参数设定优化器
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()), lr=opt.train.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(itertools.chain(netD_B.parameters(), netD_A.parameters()), lr=opt.train.lr, betas=(0.5, 0.999))

# 优化器的学习率调整策略
schedulers_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.train.total_epochs, eta_min=0)
schedulers_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=opt.train.total_epochs, eta_min=0)

# 图像增强的评价指标
train_evaluator = evaluate.ImageEnhanceEvaluator()

# 临时变量, 判断是真是假, 不必每次临时分配(换判别器的话, 这里也要换)
discrim_size = tuple(netD_A(torch.randn(1, 3, *opt.datasets.low_res).cuda()).shape[-2:])
TRUE = torch.ones(opt.train.train_batch_size, 1, *discrim_size).cuda()
FALSE = torch.zeros(opt.train.train_batch_size, 1, *discrim_size).cuda()

# 记录最佳的 epoch
best_epoch = 0
best_psnr = -1e3

# 开始训练
for epoch in range(1, opt.train.total_epochs + 1):    
    
    # 准备
    schedulers_G.step()
    schedulers_D.step()
    train_evaluator.clear()
    train_loss_logger = evaluate.LossLogger()
    
    for train_batch, data in enumerate(train_loader, 1):
        # 取出数据
        real_A = data['A'].cuda()
        real_B = data['B'].cuda()
        # -------------------- 训练生成器, 暂时冻结判别器的权重 -------------------- 
        with utils.FreezeScope([netD_B, netD_A]) as train_generator:
            # 清空 G 优化器的梯度
            optimizer_G.zero_grad()
            # A -> B
            fake_B = netG_A2B(real_A)  
            # A -> B -> A
            fake_cycle_A = netG_B2A(fake_B)  
            # B -> A
            fake_A = netG_B2A(real_B) 
            # B -> A -> B
            fake_cycle_B = netG_A2B(fake_A)  
            # 对 fake_B 和 fake_A 判别真假
            fake_B_guess = netD_B(fake_B)
            fake_A_guess = netD_A(fake_A)
            # 生成器希望得到的图片可以骗过判别器, 被认定为真, 1
            loss_G_A2B = loss_adversarial(fake_B_guess, TRUE)
            loss_G_B2A = loss_adversarial(fake_A_guess, TRUE)
            # Cycle 重建的图像, 尽量和原图一致, L1loss
            loss_cycle_A2B2A = loss_cycle(fake_cycle_A, real_A)
            loss_cycle_B2A2B = loss_cycle(fake_cycle_B, real_B)
            # 计算总损失
            loss_G_value = opt.train.weight_adversarial * loss_G_A2B + loss_G_B2A \
                         + opt.train.weight_cycle * (loss_cycle_A2B2A + loss_cycle_B2A2B)
            # + identity loss, 个人以为是为前向添加强约束, 得到的结果不会太离谱 ?
            if(opt.train.add_identity):
                identity_A2B = netG_A2B(real_B)
                identity_B2A = netG_B2A(real_A)
                loss_G_value += opt.train.weight_identity * (loss_identity(identity_A2B, real_B) + loss_identity(identity_B2A, real_A))
            # backward, 计算梯度
            loss_G_value.backward()     
            # 更新生成器的权重
            optimizer_G.step()    

        # -------------------- 训练判别器, 暂时冻结生成器的权重(其实没必要, 因为下面计算用到的 tensor 跟生成器构不成图, fake_B 跟 fake_A 都 detach 脱离了) -------------------- 
        with utils.FreezeScope([netG_A2B, netG_B2A]) as train_discriminator:

            # 开始训练判别器
            optimizer_D.zero_grad()   
            # --------- 【1】 训练判别器 B
            # 从前若干个 batch 历史 B 生成图像中选一个
            fake_B_history = fake_B_pool.query(fake_B)
            # 判别器 B 对 real_B 和 fake_B_history 判定
            pred_real = netD_B(real_B)
            pred_fake = netD_B(fake_B_history.detach())
            # 判别器 B 认为 real_B 原图是真的, 1
            loss_D_real = loss_adversarial(pred_real, TRUE)
            # 判别器 B 认为 fake_B_history 原图是假的, 0
            loss_D_fake = loss_adversarial(pred_fake, FALSE)
            # 计算训练判别器 B 的总损失
            loss_D_B = (loss_D_real + loss_D_fake) * opt.train.weight_discriminator
            # backward, 计算梯度
            loss_D_B.backward()
            # --------- 【2】 训练判别器 B2A
            # 从前若干个 batch 历史 A 生成图像中选一个
            fake_A_history = fake_A_pool.query(fake_A)
            # 判别器对 real_A 和 fake_A_history 判定
            pred_real = netD_A(real_A)
            pred_fake = netD_A(fake_A_history.detach())
            # 判别器 A 认为 real_A 原图是真的, 1
            loss_D_real = loss_adversarial(pred_real, TRUE)
            # 判别器 A 认为 fake_A_history 原图是假的, 0
            loss_D_fake = loss_adversarial(pred_fake, FALSE)
            # 计算训练判别器 A 的总损失
            loss_D_A = (loss_D_real + loss_D_fake) * opt.train.weight_discriminator
            # backward, 计算梯度
            loss_D_A.backward()
            
            # 因为判别器 A2B 和 B2A 是同一个优化器, 所以放到最后面一起更新
            optimizer_D.step() 

        # -------------------- 可视化和信息输出 -------------------- 
        if(train_batch % opt.intermediate.visualize_batch == 0):
            up_image = torch.cat([data['A'][0], fake_B[0].detach().cpu(), fake_cycle_A[0].detach().cpu()], axis=-1)
            down_image = torch.cat([data['B'][0], fake_A[0].detach().cpu(), fake_cycle_B[0].detach().cpu()], axis=-1)
            composed = torch.cat([up_image, down_image], axis=-2)
            composed = (torch.clamp(composed, 0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
            save_path = os.path.join(opt.intermediate.visualize_dir, 'epoch_{}_batch_{}.png'.format(epoch, train_batch))
            print("saved images to===>  {}".format(save_path))
            cv2.imwrite(save_path, composed)

        if(train_batch == 200): break

        train_loss_logger.update(loss_G_A2B.item(), loss_G_B2A.item(), loss_D_B.item(), loss_D_A.item(), loss_cycle_A2B2A.item(), loss_cycle_B2A2B.item())

        sys.stdout.write('\rTrain==> [epoch {}/{}] [batch {}/{}] [G_A2B {:.3f}] [G_B2A {:.3f}] [D_B {:.3f}] [D_A {:.3f}] [A2B2A {:.3f}] [B2A2B {:.3f}]'.format(
            epoch, opt.train.total_epochs, train_batch, len(train_loader), *train_loss_logger.get()))
    
    print('')
    if(epoch % opt.valid.valid_interval == 0):
        # A2B 网络是要测试 psnr 的, 所以这里要 eval()
        netG_A2B.eval()
        valid_evaluator = evaluate.ImageEnhanceEvaluator()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                real_A = data['A'].cuda()
                enhanced = netG_A2B(real_A)
                enhanced = torch.clamp(enhanced, 0, 1)
                valid_evaluator.update(data['B'].cuda(), enhanced)
                sys.stdout.write('\rValid==> [epoch {}/{}] [batch {}/{}] [loss {:.3f}] [mse {:.3f}] [psnr {:.3f}]'.format(
                    epoch, opt.train.total_epochs, i, opt.valid.repeat * len(valid_loader), *valid_evaluator.get()))
            # 记录最佳的成绩
            valid_psnr = valid_evaluator.get()[-1]
            if(best_psnr < valid_psnr):
                best_psnr = valid_psnr
                best_epoch = epoch
        # 保存模型
        if(epoch % opt.intermediate.save_interval == 0):
            torch.save({
                "netG_A2B": netG_A2B.state_dict(), 
                "netG_B2A": netG_B2A.state_dict(), 
                "netD_B": netD_B.state_dict(),
                "netD_A": netD_A.state_dict()}, 
                os.path.join(opt.intermediate.checkpoints_dir, "cyclegan_epoch_{}_{:.3f}.pth".format(epoch, valid_psnr)))
        print()



print('training is over !')
print("best model is {:.3f}db of PSNR at {}th epoch".format(best_psnr, best_epoch))
