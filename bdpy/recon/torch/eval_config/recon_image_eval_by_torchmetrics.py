'''Evaluate the reconstructed images on dnn feature.
!! Please run this script on GPU server.
'''


import argparse
from glob import glob
from itertools import product
import os
import copy
import hdf5storage

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

import bdpy
from bdpy.evals.metrics import profile_correlation, pattern_correlation, pairwise_identification
from bdpy.dl.torch import FeatureExtractor
from bdpy.dl.torch.models import VGG19
from bdpy.dl.torch.models import AlexNet, layer_map

#see also https://torchmetrics.readthedocs.io/en/stable/pages/all-metrics.html
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


# Main #######################################################################

def recon_image_eval_dnn(
        recon_image_dir,
        true_image_dir,
        output_file='quality_torchmetrics.pkl.gz',
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
        recon_eval_encoder='AlexNet',
        device='cuda:0'
):
    '''Reconstruction evaluation with DNN features.''
    Input:
    - recon_image_dir
    - true_image_dir
    Output:
    - output_file
    Parameters:
    - subjects
    - rois
    - recon_eval_encoder
    - recon_image_ext
    - true_image_ext
    - device
    '''

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Reconstructed image dir: {}'.format(recon_image_dir))
    print('True images dir:         {}'.format(true_image_dir))
    print('')
    print('Evaluation encoder: {}'.format(recon_eval_encoder))
    print('')

    # Loading data ###########################################################

    # Get recon image size
    reference_recon_image = glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*.' + recon_image_ext))[0]
    if not os.path.exists(reference_recon_image):
        raise RuntimeError("Not found:", reference_recon_image)

    img = Image.open(reference_recon_image)
    recon_image_size = img.size

    # True images
    true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
    if len(true_image_files) == 0:
        raise RuntimeError("Not found true images:", os.path.join(true_image_dir, '*.' + true_image_ext))

    # Load true images
    true_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in true_image_files
    ]
    true_images = []
    for f in true_image_files:
        a_img = Image.open(f).convert("RGB").resize(recon_image_size, Image.LANCZOS)
        true_images.append(a_img)


    # Evaluating reconstruiction performances ################################


    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
              'subject',  'roi', "ssim", "fid",
               "inception_score_mean", "inception_score_std",
               "lipips_val"
        ])

    for subject, roi in product(subjects, rois):
        print('DNN: {} - Subject: {} - ROI: {}'.format(recon_eval_encoder, subject, roi))

        if len(perf_df.query('subject == "{}" and roi == "{}"'.format( subject, roi))) > 0:
            print('Already done. Skipped.')
            continue

        recon_image_files = sorted(glob(os.path.join(
            recon_image_dir, subject, roi, '*normalized*.' + recon_image_ext
        )))
        recon_image_labels = [
            os.path.splitext(os.path.basename(a))[0]
            for a in recon_image_files
        ]
        # matching true and reconstructed images
        # TODO: better way?
        if len(recon_image_files) != len(true_image_files):
            raise RuntimeError('The number of true ({}) and reconstructed ({}) images mismatch'.format(
                len(true_image_files),
                len(recon_image_files)
            ))
        for tf, rf in zip(true_image_labels, recon_image_labels):
            if not tf in rf:
                raise RuntimeError(
                    'Reconstructed image for {} not found'.format(tf)
                )

        # Load reconstructed images
        recon_images = []
        for f in recon_image_files:
            a_img = Image.open(f).convert("RGB") # No need to resize
            recon_images.append(a_img)

        # prepare for torch metric
        recon_images_np = np.array([np.asarray(img) for img in recon_images])
        true_images_np = np.array([np.asarray(img) for img in true_images])
        recon_image_torch = torch.tensor(recon_images_np.transpose(0,3,1,2)) # [batch, channel, height, width] ranged [0-255] torch.unint8
        true_image_torch = torch.tensor(true_images_np.transpose(0,3,1,2))

        recon_torch_float =recon_image_torch.to(torch.float32) # [batch, channel, height, width] ranged [0-255] torch.float32
        true_torch_float = true_image_torch.to(torch.float32)

        recon_torch_normalize =recon_torch_float /(255/2) - 1# [batch, channel, height, width] ranged [-1, 1] torch.float32
        true_torch_normalize = true_torch_float /(255/2) - 1


        ## Calculate evaluation metrics
        ## STRUCTURAL SIMILARITY INDEX MEASURE (SSIM) https://torchmetrics.readthedocs.io/en/stable/image/structural_similarity.html

        # SSIM is a perception-based model that considers image degradation as perceived change in structural information, while also incorporating
        #important perceptual phenomena, including both luminance masking and contrast masking terms. The difference with other techniques such as MSE
        #or PSNR is that these approaches estimate absolute errors. Structural information is the idea that the pixels have strong inter-dependencies
        #especially when they are spatially close. These dependencies carry important information about the structure of the objects in the visual scene.
        #Luminance masking is a phenomenon whereby image distortions (in this context) tend to be less visible in bright regions, while contrast masking is a
        #phenomenon whereby distortions become less visible where there is significant activity or "texture" in the image.

        #from https://en.wikipedia.org/wiki/Structural_similarity
        ssim = StructuralSimilarityIndexMeasure( reduction='none')
        ssim_tensor = ssim(recon_torch_float,  true_torch_float)
        ssim_val = ssim_tensor.cpu().detach().numpy() #[bs, ssim]

        ## FRECHET INCEPTION DISTANCE (FID) https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html

        #Calculates Fréchet inception distance (FID) which is used to access the quality of generated images. Given by
        #FID = |\mu - \mu_w| + tr(\Sigma + \Sigma_w - 2(\Sigma \Sigma_w)^{\frac{1}{2}})
        #where \mathcal{N}(\mu, \Sigma) is the multivariate normal distribution estimated from Inception v3 [1] features calculated on real life images and
        #\mathcal{N}(\mu_w, \Sigma_w) is the multivariate normal distribution estimated from Inception v3 features calculated on generated (fake) images.
        # The metric was originally proposed in [1].

        #Using the default feature extraction (Inception v3 using the original weights from [2]), the input is expected to be mini-batches of 3-channel RGB
        #images of shape (3 x H x W) with dtype uint8. All images will be resized to 299 x 299 which is the size of the original training data. The boolian flag
        #real determines if the images should update the statistics of the real distribution or the fake distribution.

        #References
        #[1] Rethinking the Inception Architecture for Computer Vision Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
        # https://arxiv.org/abs/1512.00567
        #[2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
        #Bernhard Nessler, Sepp Hochreiter https://arxiv.org/abs/1706.08500

        fid = FrechetInceptionDistance(feature=2048)
        fid.update(true_image_torch, real=True)
        fid.update(recon_image_torch, real=False)
        fid_tensor = fid.compute()
        fid_val = fid_tensor.cpu().detach().numpy() #[1] smaller is better

        ## INCEPTION SCORE https://torchmetrics.readthedocs.io/en/stable/image/inception_score.html
        # Is it useful for evaluating reconstructed images?

        #Calculates the Inception Score (IS) which is used to access how realistic generated images are. It is defined as
        # IS = exp(\mathbb{E}_x KL(p(y | x ) || p(y)))
        #where KL(p(y | x) || p(y)) is the KL divergence between the conditional distribution p(y|x) and the margianl distribution p(y). Both the conditional and
        #marginal distribution is calculated from features extracted from the images. The score is calculated on random splits of the images such that both a
        #mean and standard deviation of the score are returned. The metric was originally proposed in [1].

        #Using the default feature extraction (Inception v3 using the original weights from [2]), the input is expected to be mini-batches of 3-channel RGB
        #images of shape (3 x H x W) with dtype uint8. All images will be resized to 299 x 299 which is the size of the original training data.

        #References
        #[1] Improved Techniques for Training GANs Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, Xi Chen
        #https://arxiv.org/abs/1606.03498
        #[2] GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium, Martin Heusel, Hubert Ramsauer, Thomas Unterthiner,
        #Bernhard Nessler, Sepp Hochreiter https://arxiv.org/abs/1706.08500
        inception= InceptionScore()
        inception.update(recon_image_torch)
        inception_mean, inception_std = inception.compute() #[1] [1] The larget mean is better
        inception_mean_val = inception_mean.cpu().detach().numpy()
        inception_std_val = inception_std.cpu().detach().numpy()

        # LEARNED PERCEPTUAL IMAGE PATCH SIMILARITY (LPIPS)
        # https://torchmetrics.readthedocs.io/en/stable/image/learned_perceptual_image_patch_similarity.html
        #The Learned Perceptual Image Patch Similarity (LPIPS_) is used to judge the perceptual similarity between two images. LPIPS essentially computes
        #the similarity between the activations of two image patches for some pre-defined network. This measure has been shown to match human perseption
        #well. A low LPIPS score means that image patches are perceptual similar.

        #Both input image patches are expected to have shape [N, 3, H, W] and be normalized to the [-1,1] range. The minimum size of H, W depends on the
        #chosen backbone (see net_type arg).
        lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')
        lipips_torch = lpips(recon_torch_normalize, true_torch_normalize)#[1]  The smaller better
        lipips_val = lipips_torch.cpu().detach().numpy()

        perf_df = perf_df.append({
                'subject': subject,
                'roi':     roi,
               "ssim": ssim_val, #[bs, ssim] larget is better
               "fid": fid_val, # #[1] smaller is better
               "inception_score_mean": inception_mean_val,#[1] The larget mean is better
               "inception_score_std":  inception_std_val,
               "lipips_val": lipips_val ##[1]  The smaller is better


           }, ignore_index=True)
    print(perf_df)
    # Save the results
    perf_df.to_pickle(output_file, compression='gzip')
    print('Saved {}'.format(output_file))

    print('All done')

    return output_file




# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    parser.add_argument(
        '--extend',
        type=str,
        default= None,
        help='extended_name',
    )
    args = parser.parse_args()

    if args.extend is not None:
        extend = args.extend
    else:
        extend = None

    conf_file = args.conf

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': os.path.splitext(os.path.basename(conf_file))[0]
    })

    with open(conf['feature decoding'], 'r') as f:
        conf_featdec = yaml.safe_load(f)

    conf.update({
        'feature decoding': conf_featdec,
    })

    if 'analysis name' in conf['feature decoding']:
        analysis_name = conf['feature decoding']['analysis name']
    else:
        analysis_name = ''

    if extend is None:
        recon_image_dir = os.path.join(conf['recon output dir'], analysis_name)
    else:
        recon_image_dir = os.path.join(conf['recon output dir'], extend + analysis_name)
    if extend is None:
        output_file=os.path.join(conf['recon output dir'], analysis_name,'quality_torchmetrics.pkl.gz',)
    else:
        output_file=os.path.join(conf['recon output dir'], extend + analysis_name, 'quality_torchmetrics.pkl.gz',)

    recon_image_eval_dnn(
        recon_image_dir,
        conf['true image dir'],
        output_file=output_file,
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        recon_image_ext=conf['recon image ext'],
        true_image_ext=conf['true image ext'],
        recon_eval_encoder=conf['recon eval encoder'],
        device='cuda:0'
    )