'''Make figures of feature decoding results.'''


import argparse
from glob import glob
from itertools import product
import os
from pathlib import Path

import hdf5storage
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from PIL import Image
import yaml
from tqdm.auto import tqdm

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

def recon_image_eval(
        recon_image_dir,
        true_image_dir,
        cand_image_dir,
        output_file='./quality.pkl.gz',
        subjects=[], rois=[],
        recon_image_ext='tiff',
        true_image_ext='JPEG',
        same_candidates_on_batch=False,
        n_candidates=10, n_iterations=100,
):

    # Display information
    print('Subjects: {}'.format(subjects))
    print('ROIs:     {}'.format(rois))
    print('')
    print('Reconstructed image dir:  {}'.format(recon_image_dir))
    print('True images dir:          {}'.format(true_image_dir))
    print('Candidate images dir:     {}'.format(cand_image_dir))
    print('Same candidates on batch: {}'.format(same_candidates_on_batch))
    print('')

    # Loading data ###########################################################

    # Get recon image size
    img = Image.open(glob(os.path.join(recon_image_dir, subjects[0], rois[0], '*normalized*.' + recon_image_ext))[0])
    recon_image_size = img.size

    # True images
    true_image_files = sorted(glob(os.path.join(true_image_dir, '*.' + true_image_ext)))
    true_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in true_image_files
    ]

    true_images = np.vstack([
        np.array(Image.open(f).resize(recon_image_size)).flatten()
        for f in true_image_files
    ])

    # Candidatea images
    cand_image_files = sorted(glob(os.path.join(cand_image_dir, '*.' + true_image_ext)))
    cand_image_labels = [
        os.path.splitext(os.path.basename(a))[0]
        for a in cand_image_files
    ]

    cand_images = np.vstack([
        np.array(Image.open(f).convert("RGB").resize(recon_image_size)).flatten()
        for f in cand_image_files
    ])

    # Evaluating reconstruiction performances ################################

    if os.path.exists(output_file):
        print('Loading {}'.format(output_file))
        perf_df = pd.read_pickle(output_file)
    else:
        print('Creating an empty dataframe')
        perf_df = pd.DataFrame(columns=[
            'subject', 'roi', 'multi-candidate identification accuracy'
        ])

    for subject, roi in product(subjects, rois):
        print('Subject: {} - ROI: {}'.format(subject, roi))

        if len(perf_df.query('subject == "{}" and roi == "{}"'.format(subject, roi))) > 0:
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
        recon_images = np.vstack([
            np.array(Image.open(f)).flatten()
            for f in recon_image_files
        ])

        # Calculate evaluation metrics

        ident = multi_way_image_identification(recon_images, true_images, cand_images,
                                               cand_num=n_candidates, iterate=n_iterations,
                                               metric='correlation',
                                               same_candidates_on_batch=same_candidates_on_batch)

        print('Mean identification accuracy: {}'.format(np.nanmean(ident)))

        perf_df = perf_df.append(
            {
                'subject': subject,
                'roi':     roi,
                'multi-candidate identification accuracy': ident.flatten(),
            },
            ignore_index=True
        )

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
    # parser.add_argument(
    #     '--extend',
    #     type=str,
    #     default= None,
    #     help='extended_name',
    # )
    args = parser.parse_args()

    conf_file = args.conf
    # if args.extend is not None:
    #     extend = args.extend
    # else:
    #     extend = None

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)

    conf.update({
        '__filename__': Path(conf_file).stem
    })

    with open(conf['feature decoding'], 'r') as f:
        conf_featdec = yaml.safe_load(f)

    conf.update({
        'feature decoding': conf_featdec
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

    output_file = os.path.join(conf['recon output dir'], analysis_name, '{}-candidates_pairwise_identification_{}.pkl.gz'.format(n_candidates, candidate_image_name))

    recon_image_eval(
        recon_image_dir,
        conf['true image dir'],
        cand_image_dir,
        output_file=output_file,
        subjects=conf['recon subjects'],
        rois=conf['recon rois'],
        same_candidates_on_batch=same_candidates_on_batch,
    )