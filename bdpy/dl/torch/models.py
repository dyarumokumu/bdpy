import torch
import torch.nn as nn
from types import MethodType

__all__ = ['layer_map', 'VGG19', 'AlexNet', 'AlexNetGenerator', 'StyleGAN3Generator']

def layer_map(net):
    maps = {
        'vgg19': {
            'conv1_1': 'features[0]',
            'conv1_2': 'features[2]',
            'conv2_1': 'features[5]',
            'conv2_2': 'features[7]',
            'conv3_1': 'features[10]',
            'conv3_2': 'features[12]',
            'conv3_3': 'features[14]',
            'conv3_4': 'features[16]',
            'conv4_1': 'features[19]',
            'conv4_2': 'features[21]',
            'conv4_3': 'features[23]',
            'conv4_4': 'features[25]',
            'conv5_1': 'features[28]',
            'conv5_2': 'features[30]',
            'conv5_3': 'features[32]',
            'conv5_4': 'features[34]',
            'fc6':     'classifier[0]',
            'relu6':   'classifier[1]',
            'fc7':     'classifier[3]',
            'relu7':   'classifier[4]',
            'fc8':     'classifier[6]',
        },

        'alexnet': {
            'conv1': 'features[0]',
            'conv2': 'features[4]',
            'conv3': 'features[8]',
            'conv4': 'features[10]',
            'conv5': 'features[12]',
            'fc6':   'classifier[0]',
            'relu6': 'classifier[1]',
            'fc7':   'classifier[2]',
            'relu7': 'classifier[3]',
            'fc8':   'classifier[4]',
        },

        'CLIP_ViT-B_32': {
            'conv1': 'conv1',
            ('transformer_resblocks0_attn_output', None):
                'transformer.resblocks[0].attn',
            'transformer_resblocks0_mlp': 'transformer.resblocks[0].mlp',
            ('transformer_resblocks1_attn_output', None):
                'transformer.resblocks[1].attn',
            'transformer_resblocks1_mlp': 'transformer.resblocks[1].mlp',
            ('transformer_resblocks2_attn_output', None):
                'transformer.resblocks[2].attn',
            'transformer_resblocks2_mlp': 'transformer.resblocks[2].mlp',
            ('transformer_resblocks3_attn_output', None):
                'transformer.resblocks[3].attn',
            'transformer_resblocks3_mlp': 'transformer.resblocks[3].mlp',
            ('transformer_resblocks4_attn_output', None):
                'transformer.resblocks[4].attn',
            'transformer_resblocks4_mlp': 'transformer.resblocks[4].mlp',
            ('transformer_resblocks5_attn_output', None):
                'transformer.resblocks[5].attn',
            'transformer_resblocks5_mlp': 'transformer.resblocks[5].mlp',
            ('transformer_resblocks6_attn_output', None):
                'transformer.resblocks[6].attn',
            'transformer_resblocks6_mlp': 'transformer.resblocks[6].mlp',
            ('transformer_resblocks7_attn_output', None):
                'transformer.resblocks[7].attn',
            'transformer_resblocks7_mlp': 'transformer.resblocks[7].mlp',
            ('transformer_resblocks8_attn_output', None):
                'transformer.resblocks[8].attn',
            'transformer_resblocks8_mlp': 'transformer.resblocks[8].mlp',
            ('transformer_resblocks9_attn_output', None):
                'transformer.resblocks[9].attn',
            'transformer_resblocks9_mlp': 'transformer.resblocks[9].mlp',
            ('transformer_resblocks10_attn_output', None):
                'transformer.resblocks[10].attn',
            'transformer_resblocks10_mlp': 'transformer.resblocks[10].mlp',
            ('transformer_resblocks11_attn_output', None):
                'transformer.resblocks[11].attn',
            'transformer_resblocks11_mlp': 'transformer.resblocks[11].mlp',
            'ln_post': 'ln_post',
            'model_output': 'model_output'
        },

        'ArcFace': {
            'input_layer': 'input_layer',
            'bottleneck_IR_SE0': 'body[0]',
            'bottleneck_IR_SE1': 'body[1]',
            'bottleneck_IR_SE2': 'body[2]',
            'bottleneck_IR_SE3': 'body[3]',
            'bottleneck_IR_SE4': 'body[4]',
            'bottleneck_IR_SE5': 'body[5]',
            'bottleneck_IR_SE6': 'body[6]',
            'bottleneck_IR_SE7': 'body[7]',
            'bottleneck_IR_SE8': 'body[8]',
            'bottleneck_IR_SE9': 'body[9]',
            'bottleneck_IR_SE10': 'body[10]',
            'bottleneck_IR_SE11': 'body[11]',
            'bottleneck_IR_SE12': 'body[12]',
            'bottleneck_IR_SE13': 'body[13]',
            'bottleneck_IR_SE14': 'body[14]',
            'bottleneck_IR_SE15': 'body[15]',
            'bottleneck_IR_SE16': 'body[16]',
            'bottleneck_IR_SE17': 'body[17]',
            'bottleneck_IR_SE18': 'body[18]',
            'bottleneck_IR_SE19': 'body[19]',
            'bottleneck_IR_SE20': 'body[20]',
            'bottleneck_IR_SE21': 'body[21]',
            'bottleneck_IR_SE22': 'body[22]',
            'bottleneck_IR_SE23': 'body[23]',
            'output_layer': 'output_layer'
        }
    }
    return maps[net]


class VGG19(nn.Module):
    def __init__(self):

        super(VGG19, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:

        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=False), 
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=False),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class AlexNetGenerator(nn.Module):

    def __init__(self, input_size=4096, n_out_channel=3, device=None):

        super(AlexNetGenerator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]

        self.defc7 = nn.Linear(input_size, 4096)
        self.relu_defc7 = nn.LeakyReLU(0.3,inplace=True)

        self.defc6 = nn.Linear(4096, 4096)
        self.relu_defc6 = nn.LeakyReLU(0.3,inplace=True)

        self.defc5 = nn.Linear(4096, 4096)
        self.relu_defc5 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv5 = nn.LeakyReLU(0.3,inplace=True)

        self.conv5_1 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3,inplace=True)

        self.conv4_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3,inplace=True)

        self.conv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv0 = nn.ConvTranspose2d(32, n_out_channel, kernel_size=4, stride=2, padding=1, bias=True)

        self.defc = nn.Sequential(
            self.defc7,
            self.relu_defc7,
            self.defc6,
            self.relu_defc6,
            self.defc5,
            self.relu_defc5,
        ).to(self.__device0)

        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.deconv1,
            self.relu_deconv1,
            self.deconv0,
        ).to(self.__device1)

    def forward(self, z):

        f = self.defc(z)
        f = f.view(-1, 256, 4, 4)
        g = self.deconv(f)

        return g

class StyleGAN3Generator(nn.Module):
    def __init__(self, generator, input_space='z', **args):
        super(StyleGAN3Generator, self).__init__()
        self.generator = generator
        self.input_space = input_space
        if self.input_space == 'w':
            self.add_w_to_image_method()
    def forward(self, input_vectors):
        if self.input_space == 'z': # (B, 512) -> (B, 3, 1024, 1024)
            return self.generator(input_vectors, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        elif self.input_space == 'w': # (B, 512) -> (B, 3, 1024, 1024)
            return self.generator.w_to_image(input_vectors)
        else:
            assert self.input_space == 'w+' # (B, 16, 512) -> (B, 3, 1024, 1024)
            return self.generator.synthesis(input_vectors, update_emas=False)
    def add_w_to_image_method(self):
        def w_to_ws(self, w, truncation_psi=1, truncation_cutoff=None):
            # Broadcast and apply truncation.
            x = w.unsqueeze(1).repeat([1, self.num_ws, 1])
            if truncation_psi != 1:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
            return x
        def w_to_image(self, w):
            ws = self.mapping.w_to_ws(w)
            return self.synthesis(ws)
        self.generator.mapping.w_to_ws = MethodType(w_to_ws, self.generator.mapping)
        self.generator.w_to_image = MethodType(w_to_image, self.generator)
    def get_average_feature(self):
        if self.input_space == 'w':
            return self.generator.mapping.w_avg.detach().cpu().numpy()
        else:
            raise ValueError('initial feature is not provided for the space {}'.format(self.input_space))