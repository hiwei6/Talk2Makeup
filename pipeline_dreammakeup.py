import torch
import numpy as np
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from typing import List, Optional, Union

@dataclass
class DreamMakeupOutput(BaseOutput):
    images: Union[List[np.ndarray], List["PIL.Image.Image"]]
    nsfw_content_detected: Optional[List[bool]]

class DreamMakeupPipeline(StableDiffusionPipeline):
    """
    DreamMakeup Pipeline
    Algorithm 1 implementation:
    1. Early-stopped DDIM Inversion
    2. Pixel-domain Guidance
    3. Reverse sampling with Interpolation
    """
    
    @torch.no_grad()
    def invert(self, latents, prompt_embeds, num_inference_steps, early_stop_step):
        """
        實作 Sec 4.1: Early-stopped DDIM Inversion
        反向推導: z_0 -> z_{t*}
        """
        # 設定 Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps
        
        # 計算截止的 index
        step_ratio = self.scheduler.config.num_train_timesteps // num_inference_steps
        stop_idx = early_stop_step // step_ratio
        
        # 取得反向時間步 (從小到大: t=0 -> t=t*)
        inverted_timesteps = list(reversed(timesteps))[:stop_idx]
        
        print(f"[Phase 1] Inverting latents to t={early_stop_step}...")
        
        current_latents = latents.clone()
        
        for i, t in enumerate(tqdm(inverted_timesteps)):
            # 1. 預測噪聲
            # 注意: 這裡的 prompt_embeds 必須是 batch_size=1 (Unconditional)
            # 並且需使用 return_dict=False 以兼容不同版本
            noise_pred = self.unet(
                current_latents, 
                t, 
                encoder_hidden_states=prompt_embeds,
                return_dict=False
            )[0]
            
            # 2. DDIM Inversion Step (Eq. 8 近似)
            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            
            # 獲取下一個 timestep 的 alpha (時間往後走 t -> t+1)
            if i < len(inverted_timesteps) - 1:
                next_t = inverted_timesteps[i+1]
                alpha_bar_next = self.scheduler.alphas_cumprod[next_t]
            else:
                # 最後一步 t*
                if (t + step_ratio) < self.scheduler.config.num_train_timesteps:
                    alpha_bar_next = self.scheduler.alphas_cumprod[t + step_ratio]
                else:
                    alpha_bar_next = torch.tensor(0.0).to(self.device)

            # Tweedie's formula for pred_x0
            pred_original_sample = (current_latents - (1 - alpha_bar_t) ** 0.5 * noise_pred) / alpha_bar_t ** 0.5
            
            # Re-encoding (Inversion step)
            current_latents = alpha_bar_next ** 0.5 * pred_original_sample + (1 - alpha_bar_next) ** 0.5 * noise_pred

        return current_latents, stop_idx

    @torch.no_grad()
    def __call__(
        self,
        source_image,           # x0 (Pixel)
        masked_image_latents,   # z'_0 = E(T(x0)) (Latent)
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        early_stop_steps: int = 250,   # t*
        lambda_interp: float = 0.15,   # lambda
        guidance_scale: float = 7.5,
        **kwargs,
    ):
        device = self.device
        
        # 1. Encode Inputs
        if isinstance(prompt, str): prompt = [prompt]
        if isinstance(negative_prompt, str): negative_prompt = [negative_prompt]
        
        # --- FIX: 使用標準 encode_prompt 替代 _encode_prompt ---
        # 這樣可以正確處理 CFG 需要的 Embeddings 拼接
        
        # A. 正常生成的 Embeddings (CFG Batch Size = 2)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, 
            device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
            negative_prompt=negative_prompt
        )
        # 拼接 [negative, positive]
        cfg_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # B. Inversion 用的 Embeddings (Unconditional, Batch Size = 1)
        # 這裡我們只取 negative part (空字串或 negative prompt) 作為 null-text inversion
        # 或者是使用空字串 ""
        inv_cond, inv_uncond = self.encode_prompt(
            "", 
            device, 
            num_images_per_prompt=1, 
            do_classifier_free_guidance=True, 
            negative_prompt=None
        )
        # Inversion 通常使用 Unconditional embedding (Batch size=1)
        inv_prompt_embeds = inv_uncond

        # Encode Source Image x0 -> z0
        source_image = source_image.to(device=device, dtype=self.vae.dtype)
        z0 = self.vae.encode(source_image).latent_dist.sample() * self.vae.config.scaling_factor

        # 2. Algorithm 1 Step 1: Early-stopped DDIM Inversion
        # 傳入 inv_prompt_embeds (Batch size 1) 以匹配 z0 (Batch size 1)
        z_t_star, start_idx = self.invert(z0, inv_prompt_embeds, num_inference_steps, early_stop_steps)
        
        # 3. Algorithm 1 Step 3: Reverse Sampling with Interpolation
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        # 從 t* 開始採樣到 0
        timesteps = self.scheduler.timesteps[len(self.scheduler.timesteps) - start_idx - 1 :]
        
        latents = z_t_star
        
        print(f"[Phase 3] Reverse sampling from t={early_stop_steps} with lambda={lambda_interp}...")
        
        for i, t in enumerate(tqdm(timesteps)):
            # Expand latents for CFG (Batch size 1 -> 2)
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            # 這裡傳入 cfg_prompt_embeds (Batch size 2)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cfg_prompt_embeds,
                return_dict=False
            )[0]
            
            # CFG
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # DDIM Step
            step_output = self.scheduler.step(noise_pred, t, latents)
            hat_z0 = step_output.pred_original_sample
            
            # --- Interpolation Guidance (Eq. 12) ---
            interpolated_z0 = (1 - lambda_interp) * hat_z0 + lambda_interp * masked_image_latents
            
            # 4. Update Latents based on Interpolated z0 (Manual DDIM Update)
            prev_t = t - self.scheduler.config.num_train_timesteps // num_inference_steps
            
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            # Handle boundary condition for prev_t < 0
            if prev_t >= 0:
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_t]
            else:
                alpha_prod_t_prev = torch.tensor(1.0).to(device)
            
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # z_{t-1} calculation
            latents = (alpha_prod_t_prev ** 0.5) * interpolated_z0 + (beta_prod_t_prev ** 0.5) * noise_pred

        # 5. Decode
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])
        
        return DreamMakeupOutput(images=image, nsfw_content_detected=None)