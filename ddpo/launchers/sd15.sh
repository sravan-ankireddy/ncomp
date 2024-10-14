export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="yuvalkirstain/pickapic_v2"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --local_rank=2 \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=128 \
  --max_train_steps=2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/raid/sa53869/datasets/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
   --output_dir="tmp-sd15"

