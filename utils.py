# Python
import os
import random
import datetime
# 3rd party
import numpy
import torch


def set_seed(seed, gpu_id):
    torch.manual_seed(seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    torch.set_default_tensor_type(torch.FloatTensor)
    return torch


# 为 torch 数据随机做准备
GLOBAL_SEED = 19980212
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)



# 计时
class Timer:
    def __init__(self, message=''):
        self.message = message

    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, trace):
        _end = datetime.datetime.now()
        print('耗时  :  {}'.format(_end - self.start))



# 可视化
def visualize_a_batch(batch_images, save_path, total_size=16):
    row = int(math.sqrt(batch_images.shape[0]))
    # tensor -> numpy
    batch_images = torch.clamp(batch_images.detach().cpu().permute(0, 2, 3, 1), 0, 1).mul(255).numpy().astype('uint8')
    # (16, 512, 512, 3) -> [4 * 512, 4 * 512, 3]
    composed_images = numpy.concatenate([numpy.concatenate([batch_images[row * i + j] for j in range(row)], axis=1) for i in range(row)], axis=0)
    cv2.imwrite(save_path, composed_images)

