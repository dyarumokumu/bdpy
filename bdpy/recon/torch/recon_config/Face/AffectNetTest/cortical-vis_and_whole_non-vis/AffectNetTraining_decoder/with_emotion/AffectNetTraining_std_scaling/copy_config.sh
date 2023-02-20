PROMPT1="categories"
PROMPT2="categories_evoking-feelings"
PROMPT3="categories_feelings-of"

# coeffs=("1_5" "1" "2" "10" "27" "100")

for prompt in $PROMPT1 $PROMPT2 $PROMPT3
do
    for coeff in 1_5 1 2 10 27 100
    do
        echo $coeff
        cp "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_categories_evoking/AffectNetTest_${coeff}_std_scaled_emotion_categories_evoking_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml" \
            "/home/eitoikuta/bdpy_update/bdpy/bdpy/recon/torch/recon_config/Face/AffectNetTest/cortical-vis_and_whole_non-vis/AffectNetTraining_decoder/with_emotion/AffectNetTraining_std_scaling/27_${prompt}/AffectNetTest_${coeff}_std_scaled_emotion_${prompt}_DA_decoded_KS_cortical_VC_CLIP_average_w-StyleGAN3_600iter.yaml"
    done
done
