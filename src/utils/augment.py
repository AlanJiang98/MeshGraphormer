import albumentations as A

class PhotometricAug(object):
    """
    Color augment
    """

    def __init__(self, config) -> None:
        self.augmentor = A.Compose([
            A.ColorJitter(p=config['colorjitter']['p'],
                          brightness=(config['colorjitter']['brightness']['thre'][0],
                                      config['colorjitter']['brightness']['thre'][1])),
            A.Blur(p=config['blur']['p'], blur_limit=(3, 6)),
            A.RandomGamma(p=config['gamma']['p'], gamma_limit=(15, 65)),
            A.GaussNoise(p=config['gauss']['p'],
                         var_limit=(config['gauss']['var'][0], config['gauss']['var'][1]))
        ], p=1.0)

    def __call__(self, x):
        return self.augmentor(image=x)['image']
