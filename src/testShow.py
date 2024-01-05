# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn

# ============================ 手动选择gpu
# flag = 0
flag = 1
if flag:
    # gpu_list=[2,3] 如果你的炼丹炉只有一个GPU，这样设置也没有用，当前设备没有2号和3号GPU，pt在运行的时候的device_count属性为0
    gpu_list = [0]  # 因此要设置为0号，这样device_count属性才会为1
    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================ 依内存情况自动选择主gpu
# flag = 0
flag = 1
if flag:
    def get_gpu_memory():
        import platform
        if 'Windows' != platform.system():
            import os
            os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
            memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
            os.system('rm tmp.txt')
        else:
            memory_gpu = False
            print("显存计算功能暂不支持windows操作系统")
            
        return memory_gpu


    gpu_memory = get_gpu_memory() # 这里是获取所有GPU的剩余内存
    if not gpu_memory:
        print("\ngpu free memory: {}".format(gpu_memory)) # 然后打印出来
        gpu_list = np.argsort(gpu_memory)[::-1]

        gpu_list_str = ','.join(map(str, gpu_list))
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str) # 这里是把剩余内存最多的GPU做为主GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FooNet(nn.Module):
    def __init__(self, neural_num, layers=3):
        super(FooNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])

    def forward(self, x):

        print("\nbatch size in forward: {}".format(x.size()[0])) # 观察每个forward的batchsize大小
        # 注意，这里是传入的batchsize经过分发后的数量，所以应该是原batchsize除以GPU的数量
        # 这里的 batch_size = 16，如果有device_count=1，这里应该是16，如果是device_count=2，这里应该是8.
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            x = torch.relu(x)
        return x


if __name__ == "__main__":

    batch_size = 16

    # data
    inputs = torch.randn(batch_size, 3)
    labels = torch.randn(batch_size, 3)

    inputs, labels = inputs.to(device), labels.to(device) # 把输入和标签放到指定的device中，device根据上面的代码是优先GPU的。

    # model
    net = FooNet(neural_num=3, layers=3)
    net = nn.DataParallel(net) # 对模型进行包装，使得模型具有并行分发运行的能力，让模型能把一个batchsize的数据分发到不同GPU上进行运算
    net.to(device)

    # training
    for epoch in range(1):

        outputs = net(inputs)

        print("model outputs.size: {}".format(outputs.size()))

    print("CUDA_VISIBLE_DEVICES :{}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("device_count :{}".format(torch.cuda.device_count()))




