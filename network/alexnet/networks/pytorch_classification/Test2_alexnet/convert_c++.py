import torch
import torchvision
from PIL import Image
import numpy as np
from model import AlexNet

# 图片发在了build文件夹下
image = Image.open("/home/zhongsy/datasets/dataset/train/no_obstacle/0.jpg")
image = image.resize((224, 224), Image.ANTIALIAS)
image = np.asarray(image)
image = image / 255
image = torch.Tensor(image).unsqueeze_(dim=0)
image = image.permute((0, 3, 1, 2)).float()

model = torch.load('./AlexNet.pt',map_location=torch.device('cpu'))
model.eval()
input_cpu_ = image.cpu()
input_gpu = image.cuda()


torchd_cpu = torch.jit.trace(model, input_cpu_)
torch.jit.save(torchd_cpu, "cpu.pth")

model_gpu = torch.load('./AlexNet.pt')
model_gpu.eval()

torchd_gpu = torch.jit.trace(model_gpu,input_gpu)
torch.jit.save(torchd_gpu, "gpu.pth")
