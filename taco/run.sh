# python -u train.py --dist_port 6412 --train_dataset_root_path /scratch/09004/sravana/MSCOCO --lpips_coefficient 0.1 --joint_image_text_loss_coefficient 0.0005 --epochs 50 --num-workers 12 --lambda 0.00001 --batch-size 24 --patch-size 256 256 --seed 100 --clip_max_norm 1.0 --lr_epoch 45 48 

python -u train.py --dist_port 6412 --train_dataset_root_path /scratch/09004/sravana/MSCOCO --lpips_coefficient 3.5 --joint_image_text_loss_coefficient 0.0025 --epochs 10 --num-workers 12 --lambda 0.0004 --batch-size 20 --patch-size 256 256 --seed 100 --clip_max_norm 1.0 --lr_epoch 5 8 --checkpoint /work/09004/sravana/ls6/ncomp/taco/checkpoint_afa_v3/exp_lambda_0.0004_seed_100.0_batch_size_20_joint_image_text_loss_coefficient_0.0025_lpips_coefficient_3.5/recent_model_PSNR_27.63936_MS_SSIM_0.92093_BPP_0.15746_LPIPS_0.02783_epoch_49.pth.tar