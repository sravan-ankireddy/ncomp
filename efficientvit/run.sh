# # generate latents
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 --master-port=6001 -m applications.dc_ae.dc_ae_generate_latent resolution=512 \
#     image_root_path=/raid/sa53869/datasets/imagenet/train batch_size=64 \
#     model_name=dc-ae-f32c32-in-1.0 scaling_factor=0.2889 \
#     latent_root_path=/raid/sa53869/datasets/imagenet/latent/dc_ae_f32c32_in_1.0/imagenet_512

# # generate reference for FID computation
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node=1 --master-port=6002 -m applications.dc_ae.generate_reference \
#     dataset=imagenet imagenet.resolution=512 imagenet.image_mean=[0.,0.,0.] imagenet.image_std=[1.,1.,1.] split=train \
#     fid.save_path=/raid/sa53869/datasets/imagenet/fid/imagenet_512_train.npz

# Train the LDM model: DiT or UViT f64c128
CUDA_VISIBLE_DEVICES=2,3 torchrun --nnodes 1 --nproc_per_node=2 --master-port=6003 -m applications.dc_ae.train_dc_ae_diffusion_model resolution=512 \
    train_dataset=latent_imagenet latent_imagenet.batch_size=128 latent_imagenet.data_dir=/raid/sa53869/datasets/imagenet/latent/dc_ae_f64c128_in_1.0/imagenet_512 \
    evaluate_dataset=sample_class sample_class.num_samples=50000 \
    autoencoder=dc-ae-f64c128-in-1.0 scaling_factor=0.2889 \
    model=uvit uvit.depth=28 uvit.hidden_size=1152 uvit.num_heads=16 uvit.in_channels=448 uvit.patch_size=1 \
    uvit.train_scheduler=DPM_Solver uvit.eval_scheduler=DPM_Solver \
    optimizer.name=adamw optimizer.lr=2e-4 optimizer.weight_decay=0.03 optimizer.betas=[0.99,0.99] lr_scheduler.name=constant_with_warmup lr_scheduler.warmup_steps=5000 amp=bf16 \
    max_steps=500000 ema_decay=0.9999 \
    fid.ref_path=/raid/sa53869/datasets/imagenet/fid/imagenet_512_train.npz \
    run_dir=/raid/sa53869/efficientvit/exp_comp/diffusion/imagenet_512/dc_ae_f64c128_in_1.0/uvit_h_1/bs_1024_lr_2e-4_bf16 log=False
    

# # Train the LDM model: DiT or UViT f32c32
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node=1 --master-port=6003 -m applications.dc_ae.train_dc_ae_diffusion_model resolution=512 \
#     train_dataset=latent_imagenet latent_imagenet.batch_size=128 latent_imagenet.data_dir=/raid/sa53869/datasets/imagenet/latent/dc_ae_f32c32_in_1.0/imagenet_512 \
#     evaluate_dataset=sample_class sample_class.num_samples=50000 \
#     autoencoder=dc-ae-f32c32-in-1.0 scaling_factor=0.2889 \
#     model=uvit uvit.depth=28 uvit.hidden_size=128 uvit.num_heads=16 uvit.in_channels=352 uvit.patch_size=1 \
#     uvit.train_scheduler=DPM_Solver uvit.eval_scheduler=DPM_Solver \
#     optimizer.name=adamw optimizer.lr=2e-4 optimizer.weight_decay=0.03 optimizer.betas=[0.99,0.99] lr_scheduler.name=constant_with_warmup lr_scheduler.warmup_steps=5000 amp=bf16 \
#     max_steps=500000 ema_decay=0.9999 \
#     fid.ref_path=/raid/sa53869/datasets/imagenet/fid/imagenet_512_train.npz \
#     run_dir=/raid/sa53869/efficientvit/exp/diffusion/imagenet_512/dc_ae_f32c32_in_1.0/uvit_h_1/bs_1024_lr_2e-4_bf16 log=False
 

# full DC-AE-Diffusion model list: https://huggingface.co/collections/mit-han-lab/dc-ae-diffusion-670dbb8d6b6914cf24c1a49d

# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node=1 --master-port=6004 -m applications.dc_ae.eval_dc_ae_diffusion_model dataset=imagenet_512 model=dc-ae-f64c128-in-1.0-uvit-h-in-512px cfg_scale=1.0 run_dir=tmp
# Expected results:
#   fid: 13.754458694549271