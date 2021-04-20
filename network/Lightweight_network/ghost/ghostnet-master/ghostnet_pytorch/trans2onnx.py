import torch
from torch.autograd import Variable
import onnx
from ghostnet import ghostnet


print(torch.__version__)

input_name = ['input']
# output_name = ["output_conv10,""output_mask"]
output_name = ['output']

batch_size = 1
input_shape = (3, 224, 224)

input_data = torch.randn(batch_size, *input_shape)
input_data = input_data.cuda()

model = torch.load('./ghost.pt').cuda()
model.eval()

torch.onnx.export(model, input_data, 'cgu_ghost.onnx', input_names=input_name,
                  output_names=output_name, verbose=True)
