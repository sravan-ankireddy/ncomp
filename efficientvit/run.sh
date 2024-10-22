# generate latents
torchrun --nnodes 1 --nproc_per_node=1 --master-port=29501 -m applications.dc_ae.dc_ae_generate_latent resolution=512 \
    image_root_path=/raid/sa53869/datasets/imagenet/ILSVRC/Data/CLS-LOC/train batch_size=64 \
    model_name=dc-ae-f32c32-in-1.0 scaling_factor=0.2889 \
    latent_root_path=/raid/sa53869/datasets/imagenet/latent/dc_ae_f32c32_in_1.0/imagenet_512