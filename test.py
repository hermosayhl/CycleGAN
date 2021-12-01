# Python
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# 3rd party
import cv2
import yaml
import torch
import torch.nn as nn
# self
import utils
import pipeline
import evaluate
import architectures


# 命令行参数
default_yaml = "./options/test/test_fivek_unpaired.yaml"
if(len(sys.argv) < 3 or not os.path.exists(sys.argv[2])):
    yaml_path = default_yaml
else:
    yaml_path = sys.argv[2]
    

# 读取配置文件
with open(yaml_path, 'r') as yaml_reader:
    config = yaml.load(yaml_reader, Loader=yaml.FullLoader)
    opt = utils.dict2object(config)

# 随便一个简单环境
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.common.gpu_id)
torch.set_num_threads(4)
torch.set_default_tensor_type(torch.FloatTensor)

# 获取数据
train_images_list, valid_images_list, test_images_list = pipeline.get_unpaired_training_data(opt)

# 构造数据读取流
test_dataset = pipeline.AlignedDataset(valid_images_list, mode='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.test.test_batch_size, shuffle=False)

# 设定网络结构
netG_A2B = architectures.Generator()

# 加载权重
netG_A2B.load_state_dict(torch.load(opt.model.netG_A2B_path)["netG_A2B"])
print("loaded weights from {}".format(opt.model.netG_A2B_path))

# 网络送到 GPU
use_cuda = opt.common.use_cuda and torch.cuda.is_available()
if(use_cuda): netG_A2B = netG_A2B.cuda()

# 默认都是 train 模式
netG_A2B.eval()

# 图像增强的评价指标
test_evaluator = evaluate.ImageEnhanceEvaluator(psnr_only=False)

output_dir = os.path.join(opt.intermediate.output_dir, opt.name)
os.makedirs(output_dir, exist_ok=True)

# 开始测试
with torch.no_grad():
    for test_batch, data in enumerate(test_loader):
        real_A = data['A']
        if(use_cuda): real_A = real_A.cuda()
        enhanced = netG_A2B(real_A)
        enhanced = torch.clamp(enhanced, 0, 1)
        test_evaluator.update(data['B'].cuda(), enhanced)
        if(opt.intermediate.save == True):
            cv2.imwrite(os.path.join(output_dir, os.path.split(data['A_paths'][0])[-1]), test_dataset.restore(enhanced.squeeze(0)))
        sys.stdout.write('\rTest==> [batch {}/{}] [loss {:.3f}] [mse {:.3f}] [psnr {:.3f}] [ssim {:.4f}]'.format(
            test_batch, len(test_loader), *test_evaluator.get()))