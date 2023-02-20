PROMPT1="categories"
PROMPT2="categories_evoking"
PROMPT3="categories_evoking-feelings"
PROMPT4="categories_feelings-of"

# coeffs=("1_5" "1" "2" "10" "27" "100")

# for coeff in 1_5 1 2 10 27 100
# do
#     python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_categories/FaceTest_${coeff}_std_scaled_emotion_categories_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
# done

# for prompt in $PROMPT1 $PROMPT2 $PROMPT3 $PROMPT4
# do
#     for coeff in 1_5 1 2 10 27 100
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#         --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${prompt}/FaceTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/without_emotion/FaceTest_train-scaling_decoded_KS_cortival-vis_CLIP_average_w-StyleGAN3_600iter.yaml"

# for prompt in $PROMPT2 $PROMPT4
# do
#     for coeff in 1_5 10
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#         --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${prompt}/FaceTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done

# for prompt in $PROMPT2 $PROMPT3 $PROMPT4
# do
#     for coeff in 27 100
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#         --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${prompt}/FaceTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/without_emotion/AffectNetTraining_std_scaling/AffectNetTest_train-scaling_decoded_KS_cortival-vis_CLIP_average_w-StyleGAN3_600iter.yaml"

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_categories_evoking/AffectNetTest_10_std_scaled_emotion_categories_evoking_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_categories_evoking/AffectNetTest_27_std_scaled_emotion_categories_evoking_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_categories_evoking/AffectNetTest_100_std_scaled_emotion_categories_evoking_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"

# for prompt in $PROMPT1 $PROMPT3 $PROMPT4
# do
#     for coeff in 27 10 100
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#             --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_${prompt}/AffectNetTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done
for coeff in 10 100
do
    python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
        --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_${PROMPT4}/AffectNetTest_${coeff}_std_scaled_emotion_${PROMPT4}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
done


for coeff in 1 1_5 2
do
    python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
        --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_categories_evoking/AffectNetTest_${coeff}_std_scaled_emotion_categories_evoking_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
done

for prompt in $PROMPT1 $PROMPT3 $PROMPT4
do
    for coeff in 1_5
    do
        python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
            --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_${prompt}/AffectNetTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
    done
done

for prompt in $PROMPT1 $PROMPT3 $PROMPT4
do
    for coeff in 1 2
    do
        python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
            --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_${prompt}/AffectNetTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
    done
done

# canceled
# for prompt in $PROMPT3 $PROMPT4
# do
#     for coeff in 27 100
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#         --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${prompt}/FaceTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done

# for coeff in 27 100
# do
#     python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${PROMPT4}/FaceTest_${coeff}_std_scaled_emotion_${PROMPT4}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
# done

# python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#     --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${PROMPT2}/FaceTest_100_std_scaled_emotion_${PROMPT2}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"

# for prompt in $PROMPT2 $PROMPT4
# do
#     for coeff in 1 2
#     do
#         python /home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_icnn_using_class.py\
#         --recon_conf "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/FaceTest/cortical-vis_and_whole_non-vis/with_emotion/std_scaling/27_${prompt}/FaceTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
#     done
# done
