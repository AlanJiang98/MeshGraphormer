import albumentations as A
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

# class PhotometricAug(object):
#     """
#     Color augment
#     """
#
#     def __init__(self, config, is_train=True) -> None:
#         self.augmentor = A.Compose([
#             A.ColorJitter(p=config['colorjitter']['p'],
#                           brightness=(config['colorjitter']['brightness']['thre'][0],
#                                       config['colorjitter']['brightness']['thre'][1])),
#             # A.Blur(p=config['blur']['p'], blur_limit=(3, 6)),
#             A.AdvancedBlur(blur_limit=(41, 51), p=1),
#             A.RandomGamma(p=config['gamma']['p'], gamma_limit=(15, 65)),
#             A.GaussNoise(p=config['gauss']['p'],
#                          var_limit=(config['gauss']['var'][0], config['gauss']['var'][1]))
#         ], p=1.0 if is_train else 0.0)
#
#     def __call__(self, x):
#         return self.augmentor(image=x)['image']


class PhotometricAug(object):
    def __init__(self, config, is_train):
        self.conifg = config
        self.is_train = is_train
        self.colorjitter_aug = [
            config['colorjitter']['p'],
            transforms.ColorJitter(
                brightness=(config['colorjitter']['brightness']['thre'][0],
                        config['colorjitter']['brightness']['thre'][1])
            )
        ]
        self.gaussianblur_aug = [
            config['gaussianblur']['p'],
            transforms.GaussianBlur(
                kernel_size=config['gaussianblur']['kernel_size'],
                sigma=config['gaussianblur']['sigma'],
            )
        ]

    @staticmethod
    def gaussian_noise(x, var_limit):
        var = (torch.rand(1) * var_limit[1]-var_limit[0]) + var_limit[0]
        noise = torch.randn_like(x) * var
        return x + noise

    @staticmethod
    def salt_pepper_noise(x, rate):
        channel = (torch.rand(1) > 0.5).sum()
        index = (torch.rand_like(x) < rate)[0]
        x_ = x.clone()
        x_[channel] += index * torch.rand_like(x)[0]
        return torch.clip(x_, 0, 1.)

    def __call__(self, x):
        if not self.is_train:
            return x
        else:
            if torch.rand(1) < self.colorjitter_aug[0]:
                x = self.colorjitter_aug[1](x)
            if torch.rand(1) < self.gaussianblur_aug[0]:
                x = self.gaussianblur_aug[1](x)
            if torch.rand(1) < self.conifg['gauss']['p']:
                x = PhotometricAug.gaussian_noise(x, var_limit=self.conifg['gauss']['var'])
            if torch.rand(1) < self.conifg['salt_pepper']['p']:
                x = PhotometricAug.salt_pepper_noise(x, self.conifg['salt_pepper']['rate'])
            return x