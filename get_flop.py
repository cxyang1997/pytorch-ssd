from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
import torch
from torchvision.models import resnet50
# from thop import profile
from torchvision.models import vgg16
from torchvision.models import vgg19
from torchvision.models import alexnet
from torchvision.models import detection
from model_summary import model_summary

output_f = open('SSD-vgg16_structure.txt', 'w')
# Classification
# model = resnet50(pretrained=True)
# model = alexnet(pretrained=True)
# model = vgg16(pretrained=True)

# Detection
# model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
# precision = 'fp32'
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
input = torch.randn(1, 3, 300, 300)
# flops, params = profile(model, inputs=(input, ))
# print(flops, params)

net = create_vgg_ssd(21, is_test=True)
print(model_summary(net, (3, 300, 300)))

output_f.write(str(model_summary(model, (3, 300, 300))))
