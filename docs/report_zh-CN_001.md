Xxxx




## ***LMDeploy*** 量化方法

    export HF_MODEL=your_model_path
    export WORK_DIR=your_save_path

    lmdeploy lite auto_awq \
    $HF_MODEL \
    --calib-dataset 'ptb' \
    --calib-samples 128 \
    --calib-seqlen 2048 \
    --w-bits 4 \
    --w-group-size 128 \
    --work-dir $WORK_DIR