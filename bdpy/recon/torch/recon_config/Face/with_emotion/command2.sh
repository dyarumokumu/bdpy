set -e

python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/constant_scaling/27_categories/FaceTest_100_constant_scaled_emotion_categories_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml
python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/constant_scaling/27_categories/FaceTest_1_4_constant_scaled_emotion_categories_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml
python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/std_scaling/27_categories_evoking/FaceTest_10_std_scaled_emotion_categories-evoking_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml
python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/std_scaling/27_categories_evoking/FaceTest_27_std_scaled_emotion_categories-evoking_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml
python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/std_scaling/27_categories_evoking-feelings/FaceTest_27_std_scaled_emotion_categories-evoking-feelings_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml
python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py --recon_conf /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/with_emotion/std_scaling/27_categories_feelings-of/FaceTest_27_std_scaled_emotion_categories-feelings-of_DA_decoded_KS_VC_CLIP_average_w-StyleGAN3_600iter.yaml