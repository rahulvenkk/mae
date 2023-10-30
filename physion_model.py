import numpy as np
import torch
from physion_feature_extraction.feature_extract_interface import PhysionFeatureExtractor
from physion_feature_extraction.utils import DataAugmentationForVideoMAE

import models_mae


def prepare_model(chkpt_dir, arch='mae_vit_large_patch16', img_size=224):
    # build model
    model = getattr(models_mae, arch)(img_size=img_size)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

class MAE(PhysionFeatureExtractor):
    def __init__(self, weights_path, model_name, patch_size, img_size=224):

        super().__init__()

        # This is an MAE model trained with pixels as targets for visualization (ViT-Large, training mask ratio=0.75)

        # download checkpoint if not exist

        chkpt_dir = weights_path

        self.img_size = img_size

        self.model_mae = prepare_model(chkpt_dir, model_name, img_size=img_size)

        self.patch_size = patch_size

        self.ps = self.img_size//(self.patch_size)

        self.ps = self.ps**2


    def transform(self, ):
        '''
        :return: Image Transform, Frame Gap, Minimum Number of Frames

        '''

        return DataAugmentationForVideoMAE(
            imagenet_normalize=True,
            rescale_size=self.img_size,
        ), 150, 4

    def extract_features(self, videos):
        '''
        videos: [B, T, C, H, W], T is usually 4 and videos are normalized with imagenet norm
        returns: [B, T, D] extracted features
        '''

        bs, t, c, h, w = videos.shape

        videos = videos.reshape(bs*t, c, h, w)

        def forward(arr, vid):

            mask = np.ones([vid.shape[0], self.ps]).astype('bool')

            mask[:, arr] = False

            mask = torch.from_numpy(mask).cuda()

            feats = self.model_mae.forward_feature(vid.cuda(), mask)[:, 1:]

            return feats

        def assign(all_feats, arr, feats):

            mask = np.ones([all_feats.shape[0], self.ps]).astype('bool')

            mask[:, arr] = False

            mask = torch.from_numpy(mask).cuda()

            all_feats[~mask] = feats.flatten(0, 1)

        arr = np.arange(self.ps)
        np.random.shuffle(arr)
        arr = np.split(arr, 4)

        feats = [forward(x, videos) for x in arr]

        all_feats = torch.zeros([feats[0].shape[0], self.ps, feats[0].shape[-1]]).to(feats[0].device)

        tmp = [assign(all_feats, ar, feat) for (feat, ar) in zip(feats, arr)]

        all_feats = all_feats.reshape(bs, t, self.ps, -1)

        return all_feats

    def extract_features_for_seg(self, videos):
        '''
        img: [B, C, H, W], Image is normalized with imagenet norm
        returns: [B, H, W, D] extracted features
        '''
        videos = videos.unsqueeze(1)

        feat = self.extract_features(videos)[:, 0]

        ps = int(np.sqrt(feat.shape[1]))
        feat = feat.view(feat.shape[0], ps, ps, -1)

        return feat

class MAE_large(MAE):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'mae_vit_large_patch16', 16)

class MAE_large_256(MAE):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'mae_vit_large_patch16', 16, 256)

class MAE_huge(MAE):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'mae_vit_huge_patch14', 14)

class MAE_base(MAE):
    def __init__(self, weights_path):
        super().__init__(weights_path, 'mae_vit_base_patch16', 16)
