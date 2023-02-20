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
from tqdm import tqdm

def multi_way_image_identification(pred, true, cand, cand_num=10, iterate=100, metric='correlation',
                                   same_candidates_on_batch=False):
    assert metric in ['correlation', 'MSE'], print('metric "{}" is not yet implemented'.format(metric))
    pred = pred.reshape(pred.shape[0], -1) # (N, d)
    true = true.reshape(true.shape[0], -1) # (N, d)
    cand = cand.reshape(cand.shape[0], -1) # (M, d)

    if metric == 'correlation':
        pred_diff = pred - np.mean(pred, axis=-1, keepdims=True) # (N, d)
        true_diff = true - np.mean(true, axis=-1, keepdims=True) # (N, d)
        cand_diff = cand - np.mean(cand, axis=-1, keepdims=True) # (M, d)
        pred_v_true = np.sum(pred_diff * true_diff, axis=-1, keepdims=True) / (np.sqrt(np.sum(pred_diff**2, axis=-1, keepdims=True)) * np.sqrt(np.sum(true_diff**2, axis=-1, keepdims=True))) # (N, 1)
        pred_diff = pred_diff[:, None, :] # (N, 1, d)
    elif metric == 'MSE':
        pred_v_true = np.mean((pred - true)**2, axis=-1)
        pred = pred[:, None, :]
    it_results = []
    for _ in tqdm(range(iterate)):
        if same_candidates_on_batch:
            rand_ind = np.tile(np.random.choice(len(cand), cand_num-1, replace=False), (len(pred), 1))
        else:
            rand_ind = np.array([np.random.choice(len(cand), cand_num-1, replace=False) for _ in range(len(pred))])

        if metric == 'correlation':
            cand_selected_diff = cand_diff[rand_ind] # (N, cand_num, d)
            corr = np.einsum('ijk,ilk->il', pred_diff, cand_selected_diff) / (np.sqrt(np.sum(pred_diff**2, axis=-1)) * np.sqrt(np.sum(cand_selected_diff**2, axis=-1)))
            # print(corr.shape) # (N, cand_num)
            scores = np.all(corr < pred_v_true, axis=-1)
        elif metric == 'MSE':
            cand_selected = cand[rand_ind] # (N, cand_num, d)
            mse = np.mean((pred - cand_selected) ** 2, axis=-1)
            # print(mse.shape) # (N, cand_num)
            scores = np.all(mse > pred_v_true, axis=-1)

        it_results.append(scores)
    return np.mean(it_results, 0)

# Main #######################################################################

def recon_image_eval_dnn(
        recon_image_dir,
        true_image_dir,
        cand_image_dir,
        output_file='quality_dnn.pkl.gz',
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
        recon_eval_encoder='AlexNet',
        device='cuda:0',
        n_candidates=10,
        n_iterations=100,
        same_candidates_on_batch=False,
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
    print('Candidate images dir:         {}'.format(cand_image_dir))
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

    # Candidatea images
    cand_image_files = sorted(glob(os.path.join(cand_image_dir, '*.' + true_image_ext)))
    cand_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in cand_image_files
    ]
    cand_images = []
    for f in cand_image_files:
        a_img = Image.open(f).convert("RGB").resize(recon_image_size, Image.LANCZOS)
        cand_images.append(a_img)

    #and_images = np.vstack([
    #    np.array(Image.open(f).convert("RGB").resize(recon_image_size)).flatten()
    #    for f in cand_image_files
    #])

    # Load DNN for metrics on DNN layer
    # We can select AlexNet or VGG19
    dnnh = DNNHandler(recon_eval_encoder, device=device)

    cand_feat = dnnh.get_activation(cand_images, flat=True)
    # Evaluating reconstruiction performances ################################


    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'dnn', 'subject', 'roi', 'layer',
            'multi-candidates feature identification accuracy'
        ])

    for subject, roi in product(subjects, rois):
        print('DNN: {} - Subject: {} - ROI: {}'.format(recon_eval_encoder, subject, roi))

        if len(perf_df.query('dnn == "{}" and subject == "{}" and roi == "{}"'.format(recon_eval_encoder, subject, roi))) > 0:
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

        # Calculate evaluation metrics
        true_feat = dnnh.get_activation(true_images, flat=True)
        recon_feat = dnnh.get_activation(recon_images, flat=True)

        for layer in dnnh.layers:
            #profile_feat = profile_correlation(recon_feat[layer], true_feat[layer])
            #pattern_feat = pattern_correlation(recon_feat[layer], true_feat[layer])
            #ident_feat   = pairwise_identification(recon_feat[layer], np.vstack([true_feat[layer],cand_feat[layer]]))

            ident_feat   = multi_way_image_identification(recon_feat[layer],true_feat[layer], cand_feat[layer],
                                                          cand_num=n_candidates, iterate=n_iterations,
                                                          metric='correlation',
                                                          same_candidates_on_batch=same_candidates_on_batch)
            print("Layer:", layer)
            #print('Mean profile correlation:     {}'.format(np.nanmean(profile_feat)))
            #print('Mean patten correlation:      {}'.format(np.nanmean(pattern_feat)))
            print('Mean identification accuracy: {}'.format(np.nanmean(ident_feat)))
            perf_df = perf_df.append({
                'dnn': recon_eval_encoder,
                'subject': subject,
                'roi':     roi,
                'layer': layer,
                'multi-candidates feature identification accuracy': ident_feat.flatten(),
            }, ignore_index=True)

    print(perf_df)
    # Save the results
    perf_df.to_pickle(output_file, compression='gzip')
    print('Saved {}'.format(output_file))

    print('All done')

    return output_file


# Class definitions ################################################################

class DNNHandler():
    """
    DNN quick handler (only forwarding)
    - AlexNet
    - VGG19
    """
    def __init__(self, encoder_name="AlexNet", device='cpu'):
        """Initialize the handler
        Parameters
        ----------
        encoder_name : str
            Specify the encoder name for the evaluation
            "AlexNet" or "VGG19".
       device : str
           Specify the machine environment.
           "cpu" or "cuda:0".
        """
        self.encoder_name = encoder_name
        self.device = device

        if encoder_name == "AlexNet":
            self.encoder = AlexNet()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/bvlc_alexnet/bvlc_alexnet.pt'
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()

            # AlexNet input image size (lab's specific value)
            self.image_size = [227, 227]
            # Mean of training images (ILSVRC2021_Training)
            self.mean_image = np.float32([104., 117., 123.])

            self.layer_mapping = layer_map("alexnet")

        elif encoder_name == "VGG19":
            self.encoder = VGG19()
            encoder_param_file = '/home/kiss/data/models_shared/pytorch/VGG_ILSVRC_19_layers/VGG_ILSVRC_19_layers.pt'
            self.encoder.to(device)
            self.encoder.load_state_dict(torch.load(encoder_param_file))
            self.encoder.eval()

            # VGG19 input image size
            self.image_size = [224, 224]
            # Mean of training images (ILSVRC2021_Training)
            self.mean_image = np.float32([104., 117., 123.])

            self.layer_mapping = layer_map("vgg19")

        else:
            raise RuntimeError("This DNN is not implemeneted in dnn_evaluator: ", self.encoder_name)

        self.layers = list(self.layer_mapping.keys())
        self.feature_extractor = FeatureExtractor(self.encoder, self.layers, self.layer_mapping, device=self.device, detach=True)

    def get_activation(self, img_obj_list, flat=False):
        '''Obtain unit activation matrix
        Parameters
        ----------
        img_obj_list : list
            The list of PIL.Image object. (OR array objects)
        flat : bool (default: False)
            If True, the extracted feature is flatten in each image.
        Returns
        -------
        dictionary
            Dict object with the extracted features.
                {
                    "layer_name": ndarray <n samples x m units> or <n samples x m channels x h units x w units >
                }
        '''
        _img_obj_list = copy.deepcopy(img_obj_list)

        if not isinstance(_img_obj_list, list):
            _img_obj_list = [_img_obj_list]
        if isinstance(_img_obj_list[0], np.ndarray):
            _img_obj_list = [Image.fromarray(a_img) for a_img in _img_obj_list]

        activations = {layer: [] for layer in self.layers}
        for a_img in _img_obj_list:
            # Resize
            a_img = a_img.resize(self.image_size, Image.LANCZOS)
            x = np.asarray(a_img)

            # DNN specific preprocessing
            if self.encoder_name in ["AlexNet", "VGG19"]:
                # Swap dimensions and colour channels
                x = np.transpose(x, (2, 0, 1))[::-1]
                # Normalization (subtract the mean image)
                x = np.float32(x) - np.reshape(self.mean_image, (3, 1, 1))

            # Get activations
            features = self.feature_extractor.run(x)
            for layer in self.layers:
                activations[layer].append(features[layer])

        # Arrange
        for layer in self.layers:
            activations[layer] = np.vstack(activations[layer])
            if flat:
                activations[layer] = activations[layer].reshape(activations[layer].shape[0], -1)

        return activations


# Entry point ################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'conf',
        type=str,
        help='analysis configuration file',
    )
    # parser.add_argument(
    #     '--extend',
    #     type=str,
    #     default= None,
    #     help='extended_name',
    # )
    args = parser.parse_args()

    # if args.extend is not None:
    #     extend = args.extend
    # else:
    #     extend = None

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

    recon_image_dir = os.path.join(conf['recon output dir'], analysis_name)

    if 'candidate image info' in conf:
        candidate_image_info = conf['candidate image info']
        if 'candidate image dir' in candidate_image_info:
            cand_image_dir = conf['candidate image dir']
        else:
            cand_image_dir = '/home/nu/data/contents_shared/ImageNetTraining/source'
        if 'candidate image name' in candidate_image_info:
            candidate_image_name = candidate_image_info['candidate image name']
        else:
            candidate_image_name = cand_image_dir.replace('/', '_')
    else:
        cand_image_dir = '/home/nu/data/contents_shared/ImageNetTraining/source'
        candidate_image_name = 'ImageNetTraining'

    if 'random seed' in conf:
        np.random.seed(conf['random seed'])
    else:
        print('seed is set to default value: 0')
        np.random.seed(0)

    same_candidates_on_batch = 'same candidates on batch' in conf and conf['same candidates on batch']
    n_candidates = conf['n_candidates'] if 'n_candidates' in conf else 10
    n_iterations = conf['n_iterations'] if 'n_iterations' in conf else 100

    output_file = os.path.join(conf['recon output dir'], analysis_name, 'dnn_{}-candidates_pairwise_identification_{}.pkl.gz'.format(n_candidates, candidate_image_name))

    recon_image_eval_dnn(
        recon_image_dir,
        conf['true image dir'],
        cand_image_dir,
        output_file=output_file,
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        recon_image_ext=conf['recon image ext'],
        true_image_ext=conf['true image ext'],
        recon_eval_encoder=conf['recon eval encoder'],
        device='cuda:0'
    )