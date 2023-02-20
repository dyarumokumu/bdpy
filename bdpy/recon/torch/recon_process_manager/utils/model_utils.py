import torch
from torchvision import transforms
from . image_processing_utils import get_image_preprocess_in_tensor_function
if __file__ == '/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_process_manager/utils/model_utils.py':
    # from importlib.machinery import SourceFileLoader
    # bdpy = SourceFileLoader("bdpy","/home/eitoikuta/bdpy_update/bdpy/bdpy/__init__.py").load_module()
    import importlib.util
    spec = importlib.util.spec_from_file_location('dl', "/home/eitoikuta/bdpy_update/bdpy/bdpy/dl/torch/models.py")
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)

    import sys
    sys.path.append('/home/eitoikuta/StyleGAN3_and_applications/inversion_and_editing/stylegan3-editing')
    from models.stylegan3.model import SG3Generator

    sys.path.append('/home/eitoikuta/facial_image_eval/identity/InsightFace_Pytorch')
    from model import Backbone
else:
    import bdpy.dl as dl

def create_model_instance(model_name, device='cpu', training=False, **args):
    if 'preprocess_info' in args:
        # TODO: specification by file
        mean = args['preprocess_info']['mean']
        std = args['preprocess_info']['std']
        if mean is not None and std is not None:
            preprocess_from_conf = get_image_preprocess_in_tensor_function(mean, std)
        else:
            preprocess_from_conf = None
    else:
        preprocess_from_conf = None

    if model_name == 'CLIP_ViT-B_32':
        import clip
        model, preprocess = clip.load('ViT-B/32', device=device)
        if not training:
            model.eval()
        if 'image_encoder_only' in args and args['image_encoder_only']:
            model = model.visual.float()
        elif 'text_encoder_only' in args and args['text_encoder_only']:
            model = model.transformer.float()
        # FIXME: the option `preprocess_from_PIL` does not work now
        if not 'preprocess_from_PIL' in args or not args['preprocess_from_PIL']:
            preprocess = transforms.Normalize((0.48145466*255, 0.4578275*255, 0.40821073*255), (0.26862954*255, 0.26130258*255, 0.27577711*255))
    elif model_name == 'AlexNetGenerator_ILSVRC2012_Training_relu7':
        # from bdpy.dl import AlexNetGenerator
        model = dl.AlexNetGenerator().to(device)
        if 'params_file' in args:
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        preprocess = None
    elif model_name == 'vgg19':
        # from bdpy.dl import VGG19
        model = dl.VGG19().to(device)
        if 'params_file' in args:
            print('loading VGG19 parameters')
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        preprocess = None
    elif model_name == 'alexnet':
        # from bdpy.dl import AlexNet
        # TODO: accept different number of classes
        model = dl.AlexNet().to(device)
        if 'params_file' in args:
            model.load_state_dict(torch.load(args['params_file']))
        if not training:
            model.eval()
        preprocess = None
    elif model_name == 'StyleGAN3':
        # models will be built on codes for StyleGAN3 editing (i.e., not the original one)
        # the pixel values of generated images range from -1 to 1. Simply (image + 1) / 2 will normalize the image into the range of [0, 1]
        generator = SG3Generator(checkpoint_path=None).decoder
        if 'params_file' in args:
            # args['params_file'] = 'pretrained_models/encoders/restyle_e4e_ffhq.pt' or 'pretrained_models/encoders/restyle_pSp_ffhq.pt'
            ckpt = torch.load(args['params_file'], map_location='cpu')
            generator.load_state_dict(_get_keys(ckpt, 'decoder', remove=["synthesis.input.transform"]), strict=False)
        if not training:
            generator.eval()
        model = dl.StyleGAN3Generator(generator, **args).to(device)
        preprocess = None
    elif model_name == 'ArcFace':
        model = Backbone(50, 0.6, 'ir_se')
        if 'params_file' in args:
            # args['param_file'] = '/home/eitoikuta/facial_image_eval/identity/InsightFace_Pytorch/work_space/models/model_ir_se50.pth'
            model.load_state_dict(torch.load(args['param_file']))
        if not training:
            model.eval()
        # preprocess = transforms.Normalize([[128, 128, 128], [128, 128, 128]])
        if not 'preprocess_from_PIL' in args or not args['preprocess_from_PIL']:
            # preprocess = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            preprocess = transforms.Normalize([128., 128., 128.], [128., 128., 128.])
        else:
            preprocess = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([128., 128., 128.], [128., 128., 128.])])
    else:
        assert False, print('Unknown model name is specified: {}'.format(model_name))

    if preprocess_from_conf is not None:
        preprocess = preprocess_from_conf
    return model, preprocess

def _get_keys(d, name, remove=[]):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items()
                if k[:len(name)] == name and k[len(name) + 1:] not in remove}
    return d_filt