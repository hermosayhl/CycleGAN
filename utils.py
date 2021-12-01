# Python
import os
import numpy
import random
import datetime
# 3rd party
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
    torch.set_num_threads(4)
    torch.set_default_tensor_type(torch.FloatTensor)
    return torch


# 为 torch 数据随机做准备
GLOBAL_SEED = 19981229
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)



def dict2object(d):  
    top = type('new', (object,), d)  
    seqs = tuple, list, set, frozenset  
    for i, j in d.items():  
        if isinstance(j, dict):  
            setattr(top, i, dict2object(j))  
        elif isinstance(j, seqs):  
            setattr(top, i,   
                type(j)(dict2object(sj) if isinstance(sj, dict) else sj for sj in j))  
        else:  
            setattr(top, i, j)  
    return top 



class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0: 
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0: 
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:  
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5: 
                    random_id = random.randint(0, self.pool_size - 1)  
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:     
                    return_images.append(image)
        return torch.cat(return_images, 0) 



# 计时
class Timer:
    def __init__(self, message=''):
        self.message = message

    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, trace):
        _end = datetime.datetime.now()
        print('耗时  :  {}'.format(_end - self.start))





# 暂时冻住某些网络的参数
class FreezeScope:
    def __init__(self, freeze_list):
        self.freeze_list = freeze_list

    def __enter__(self):
    	# 冻结
    	for some_network in self.freeze_list:
	        for param in some_network.parameters():
	            param.requires_grad = False

    def __exit__(self, type, value, trace):
        # 恢复
    	for some_network in self.freeze_list:
	        for param in some_network.parameters():
	            param.requires_grad = True