import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import torch.nn as nn

from diffusers.schedulers import EDMDPMSolverMultistepScheduler as scheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DConditionModel

from timm.layers.patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from timm.layers.pos_embed_sincos import build_sincos2d_pos_embed

from src.utils.evaluation import compute_all_metrics


def get_model(train_dataset:Dataset, load_checkpoint:bool, device, weights_path: str = 'model_weights/diffusion.pth'):

    input_tensor = train_dataset[0][0]
    img_size = input_tensor.shape[1:]
    input_channels = input_tensor.shape[-3]

    with open("config/train_model.yaml", "r") as f:
        config = yaml.safe_load(f)

    unet = config['model']['diffusion']['unet_model']
    cond_encoder = config['model']['diffusion']['condition_encoder']

    unet_config = {
        'sample_size': (80,112),
        'in_channels': input_channels,
        'out_channels': input_channels,
        'time_embedding_type': unet['time_embedding_type'],
        'flip_sin_to_cos': unet['flip_sin_to_cos'],
        'down_block_types': unet['down_block_types'],
        'up_block_types': unet['up_block_types'],
        'block_out_channels': unet['block_out_channels'],
        'act_fn': unet['act_fn'],
        'cross_attention_dim': unet['cross_attention_dim'],
        'attention_head_dim': unet['attention_head_dim']
    }

    cond_config = {
        'patch_size': cond_encoder['patch_size'],
        'in_chans': input_channels,
        'embed_dim': cond_encoder['embed_dim'],
        'depth': cond_encoder['depth'],
        'num_heads': cond_encoder['num_heads'],
        'mlp_ratio': cond_encoder['mlp_ratio']
    }

    base_model = Diffusion(img_size, unet_config, cond_config, device='cuda', in_channels=input_channels, weights_path=weights_path)

    return base_model


class ConditionEncoder(nn.Module):

    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],
        in_chans: int,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        
        # Positional embeddings
        self.positional_embedding = build_sincos2d_pos_embed(
            feat_shape=[self.grid_size[0], self.grid_size[1]],
            dim=embed_dim
        ).unsqueeze(0)
            
    
    def forward(self, x):
        """
        Forward function for the encoding part.
        """
        x = self.patch_embed(x)
        x = x + self.positional_embedding.to(x.device)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


def Unet2DCondition(config):

    return UNet2DConditionModel(**config)


class Diffusion():
    def __init__(self, img_size, unet_config, cond_config, device='cuda', in_channels=4, weights_path='model_weights/diffusion.pth'):
        super(Diffusion, self).__init__()

        self.model_path = weights_path
        self.in_chanl = in_channels

        self.cond_model = ConditionEncoder(img_size=img_size,      
                               patch_size=cond_config['patch_size'],
                                in_chans=cond_config['in_chans'],
                                embed_dim=cond_config['embed_dim'],
                                depth=cond_config['depth'],
                                num_heads=cond_config['num_heads'],
                                mlp_ratio=cond_config['mlp_ratio']
                            )
        
        self.denoiser_model = Unet2DCondition(unet_config)

        self.cond_model.to(device)
        self.denoiser_model.to(device)


    def best_model(self):
        checkpoint = torch.load(self.model_path, weights_only=False)
        self.denoiser_model.load_state_dict(checkpoint['model_state_dict'])
        self.cond_model.load_state_dict(checkpoint['cond_model_state_dict'])

    
    def _pad_input(self, x, multiple=16):
        """
        Pad input tensor (B, C, H, W) so H and W are divisible by `multiple`.
        Returns padded tensor and amount of padding applied.
        """
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        # Pad right and bottom only (left/top = 0)
        padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # pad = (left, right, top, bottom)

        return padded, pad_h, pad_w
    
    def _unpad_output(self, output, pad_h, pad_w):
        """
        Remove padding from model output (B, C, H, W)
        """
        if pad_h > 0:
            output = output[:, :, :-pad_h, :]
        if pad_w > 0:
            output = output[:, :, :, :-pad_w]
        return output

    
    def get_sigmas(self, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device='cuda'):
        # modified from diffusers/examples/dreambooth/train_dreambooth_lora_sdxl.py
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    
    def loss_fn(self, output_img, org_img, sigmas):

        alpha = (sigmas ** 2 + 0.5** 2) / (sigmas * 0.5) ** 2 
        final_loss = torch.mean((alpha * (output_img-org_img)**2))

        return final_loss
    
    def reconst_error(self, org_img, output_img):
        mse_loss = torch.nn.functional.mse_loss(org_img, output_img, reduction='mean')
        return mse_loss.item()

    
    def train_one_epoch(self, train_dataloader, optimizer, cond_optimizer, noise_sampler, noise_scheduler, device='cuda'):
            
        self.denoiser_model.train()
        self.cond_model.train()
        total_loss = 0.0
        for  vt, org_img, mask in train_dataloader:

                # Condition with concat of Voronoi tesselation

            org_img = org_img.to(device)
            mask = mask.to(device)
            vt = vt.to(device)

            optimizer.zero_grad()
            cond_optimizer.zero_grad()
            vt_cond = self.cond_model(vt)

            noise = torch.randn(org_img.shape,  device=device)
            batch_size = org_img.shape[0]

            indices = noise_sampler(batch_size, device='cpu')
            timesteps = noise_scheduler.timesteps[indices].to(device=device)
                #sigmas = noise_scheduler.sigmas[indices].to(device=device)

            noise = noise*(1-mask)
            noisy_images = noise_scheduler.add_noise(org_img, noise, timesteps)

            sigmas = self.get_sigmas(noise_scheduler, timesteps, len(noisy_images.shape), noisy_images.dtype)
            x_in = noise_scheduler.precondition_inputs(noisy_images, sigmas)

            x_in, pad_h, pad_w = self._pad_input(x_in)

            
            model_output = self.denoiser_model(x_in, timesteps, encoder_hidden_states=vt_cond, return_dict=False)[0]
            model_output = self._unpad_output(model_output, pad_h, pad_w)

            model_output = noise_scheduler.precondition_outputs(noisy_images, model_output, sigmas)
                

                # Final output loss
            loss = self.loss_fn(model_output, org_img, sigmas)
            error = self.reconst_error(org_img, model_output)
            total_loss += error

            loss.backward()
            optimizer.step()
            cond_optimizer.step()

        train_loss = total_loss/len(train_dataloader)

        return train_loss, optimizer, cond_optimizer
    


class EvaluateDiffusionModel():
    def __init__(self, model, cond_model, device='cuda'):
        self.denoiser_model = model
        self.cond_model = cond_model
        self.denoiser_model.to(device)
        self.denoiser_model.to(device)

    def _pad_input(self, x, multiple=16):
        """
        Pad input tensor (B, C, H, W) so H and W are divisible by `multiple`.
        Returns padded tensor and amount of padding applied.
        """
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        # Pad right and bottom only (left/top = 0)
        padded = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))  # pad = (left, right, top, bottom)

        return padded, pad_h, pad_w
    
    def _unpad_output(self, output, pad_h, pad_w):
        """
        Remove padding from model output (B, C, H, W)
        """
        if pad_h > 0:
            output = output[:, :, :-pad_h, :]
        if pad_w > 0:
            output = output[:, :, :, :-pad_w]
        return output
    
    def reconst_error(self, org_img, output_img):
        mse_loss = torch.nn.functional.mse_loss(org_img, output_img, reduction='mean')
        return mse_loss.item()

    def back_sampling(self, img_size, mask, vt, noise_scheduler, org_img, generator, num_inf_steps, device):

        image = randn_tensor(img_size, generator=generator, device=device, dtype=self.denoiser_model.dtype)
        noise = image.clone()

        noise_scheduler.set_timesteps(num_inf_steps)
        vt_cond = self.cond_model(vt)

        for i,t in enumerate(noise_scheduler.timesteps):
            x_in = noise_scheduler.scale_model_input(image, t)

            x_in, pad_h, pad_w = self._pad_input(x_in)

            model_output = self.denoiser_model(x_in, t, encoder_hidden_states=vt_cond, return_dict=False)[0]
            model_output = self._unpad_output(model_output, pad_h, pad_w)
            # predict Ftheta
            model_output = model_output * (1-mask) + org_img*(mask) 
            
            # Dtheta (precondition output is performed within step function)
            image = noise_scheduler.step(model_output, t, image, return_dict=False)[0]

            tmp_known_points = org_img.clone()
            if i < len(noise_scheduler.timesteps) - 1:
                noise_timestep = noise_scheduler.timesteps[i+1]
                tmp_known_points = noise_scheduler.add_noise(tmp_known_points, noise, torch.tensor([noise_timestep]))

            image = image * (1-mask) + tmp_known_points* mask

        
        return image


    def evaluate_model(self, dataloader, advanced_statistics, device='cuda'):
        self.denoiser_model.eval()
        self.cond_model.eval()

        total_loss = 0.0
        all_obs, all_gt, preds = [], [], []
        relative_errors, ssims, psnrs, local_errors = [], [], [], []

        with torch.no_grad():
            noise_scheduler_eval = scheduler(algorithm_type='sde-dpmsolver++')

            for vt, org_img, mask in dataloader:

                    # Condition with concat of Voronoi tesselation

                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)
                generator = torch.Generator(device=device).manual_seed(42)

                sample_images = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_eval, org_img, generator, num_inf_steps=50, device=org_img.device)
                    #predicted_img, _ = self.model(masked_img)

                loss = self.reconst_error(sample_images, org_img)
                total_loss+=loss

                if advanced_statistics:
                    batch_relative_error, batch_ssim, batch_psnr, batch_local_errors = compute_all_metrics(org_img, sample_images)
                    relative_errors.append(batch_relative_error)
                    ssims.append(batch_ssim)
                    psnrs.append(batch_psnr)
                    local_errors.append(batch_local_errors)

                    all_obs.append(vt.cpu().numpy())
                    all_gt.append(org_img.cpu().numpy())
                    preds.append(sample_images.cpu().numpy())

        if advanced_statistics:
            relative_errors = np.concatenate(relative_errors, axis=0)
            ssims = np.concatenate(ssims, axis=0)
            psnrs = np.concatenate(psnrs, axis=0)
            local_errors = np.concatenate(local_errors, axis=0)
            
            all_obs = np.concatenate(all_obs, axis=0)
            all_gt = np.concatenate(all_gt, axis=0)
            preds = np.concatenate(preds, axis=0)

        avg_loss = total_loss/len(dataloader)

        return avg_loss, relative_errors, ssims, psnrs, local_errors, all_obs, all_gt, preds
        

    def ensemble_prediction(self, org_img, mask, vt, num_inf_steps, num_ensem_steps, device):
        self.denoiser_model.eval()
        self.cond_model.eval()

        ensem_output_list = []
        ensemble_output = torch.zeros_like(org_img, device="cpu")
        with torch.no_grad():
            for i in range(num_ensem_steps):
                noise_scheduler_ensem = scheduler(algorithm_type='sde-dpmsolver++')
                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)

                sample_img = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_ensem, org_img, generator=None, num_inf_steps=num_inf_steps, device=org_img.device)
                ensemble_output += sample_img.cpu()
                ensem_output_list.append(ensemble_output/(i+1))

        return ensem_output_list

    def all_ensemble_output(self, org_img, mask, vt, num_inf_steps, num_ensem_steps, device):
        self.denoiser_model.eval()
        self.cond_model.eval()

        ensem_output_list = []
        with torch.no_grad():
            for i in range(num_ensem_steps):
                noise_scheduler_ensem = scheduler(algorithm_type='sde-dpmsolver++')
                org_img = org_img.to(device)
                mask = mask.to(device)
                vt = vt.to(device)

                sample_img = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_ensem, org_img, generator=None, num_inf_steps=num_inf_steps, device=org_img.device)
                ensem_output_list.append(sample_img.cpu())

        return ensem_output_list    
        

    def ensemble_inference_prediction(self, org_img, mask, vt, num_inf_steps, device):
        self.denoiser_model.eval()
        self.cond_model.eval()

        ensem_output_list = []
        with torch.no_grad():
            org_img = org_img.to(device)
            mask = mask.to(device)
            vt = vt.to(device)

            for i in range(len(num_inf_steps)):
                noise_scheduler_ensem = scheduler(algorithm_type='sde-dpmsolver++')
                    
                sample_img = self.back_sampling(org_img.shape, mask, vt, noise_scheduler_ensem, org_img, generator=None, num_inf_steps=num_inf_steps[i], device=org_img.device)
                ensem_output_list.append(sample_img.cpu())

        return ensem_output_list

