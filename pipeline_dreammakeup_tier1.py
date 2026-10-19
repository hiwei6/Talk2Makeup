"""
Tier1DreamMakeupPipeline: 以 LLM hidden states + Projector 取代 CLIP 條件化的
DreamMakeup pipeline。

與 pipeline_dreammakeup.py 的差異只有一個:條件向量的來源。
- 原版:CLIP text encoder 對 prompt 字串編碼一次,整個採樣過程用同一組 embedding。
- Tier1:Projector 是 timestep-aware 的(AdaLN 調變),所以在「每一個」去噪步 t
  都要重新呼叫 projector(hidden_states, t) 產生該步專屬的條件向量。
  DDIM Inversion 階段同理,null-text embedding 改為 projector(空字串hidden, t)。

早停 DDIM Inversion、插值引導採樣(lambda_interp)、CFG 的邏輯完全沿用原版,
顏色精準路徑(masked_image_latents = z'_0 插值錨點)也完全不變。
"""

import torch
from tqdm import tqdm

from pipeline_dreammakeup import DreamMakeupPipeline, DreamMakeupOutput


class Tier1DreamMakeupPipeline(DreamMakeupPipeline):

    def set_projector(self, projector):
        """掛上訓練好的 MakeupProjector(fp32,由呼叫端負責 .to(device).eval())。"""
        self._projector = projector

    def _embed(self, hidden, mask, t):
        """單一時間步的 Projector 呼叫:hidden [B,L,H] fp32 → [B,Q,D] 轉成 UNet dtype。"""
        t_tensor = torch.as_tensor([t], device=self.device)
        cond = self._projector(hidden, t_tensor, attention_mask=mask)
        return cond.to(self.unet.dtype)

    @torch.no_grad()
    def invert_llm(self, latents, uncond_hidden, uncond_mask,
                   num_inference_steps, early_stop_step):
        """Early-stopped DDIM Inversion,與父類 invert() 相同,
        僅將固定的 null-text embedding 換成每步呼叫 Projector。"""
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        step_ratio = self.scheduler.config.num_train_timesteps // num_inference_steps
        stop_idx = early_stop_step // step_ratio
        inverted_timesteps = list(reversed(timesteps))[:stop_idx]

        print(f"[Tier1 Phase 1] Inverting latents to t={early_stop_step}...")

        current_latents = latents.clone()

        for i, t in enumerate(tqdm(inverted_timesteps)):
            uncond_embeds = self._embed(uncond_hidden, uncond_mask, t)
            noise_pred = self.unet(
                current_latents,
                t,
                encoder_hidden_states=uncond_embeds,
                return_dict=False,
            )[0]

            alpha_bar_t = self.scheduler.alphas_cumprod[t]
            if i < len(inverted_timesteps) - 1:
                next_t = inverted_timesteps[i + 1]
                alpha_bar_next = self.scheduler.alphas_cumprod[next_t]
            else:
                if (t + step_ratio) < self.scheduler.config.num_train_timesteps:
                    alpha_bar_next = self.scheduler.alphas_cumprod[t + step_ratio]
                else:
                    alpha_bar_next = torch.tensor(0.0).to(self.device)

            pred_original_sample = (
                current_latents - (1 - alpha_bar_t) ** 0.5 * noise_pred
            ) / alpha_bar_t ** 0.5
            current_latents = (
                alpha_bar_next ** 0.5 * pred_original_sample
                + (1 - alpha_bar_next) ** 0.5 * noise_pred
            )

        return current_latents, stop_idx

    @torch.no_grad()
    def __call__(
        self,
        source_image,             # x0 (Pixel)
        masked_image_latents,     # z'_0 = E(T(x0)),顏色錨點,與原版相同
        cond_hidden,              # LLM hidden states(使用者妝容描述)[B,L,H] fp32
        cond_mask,                # 對應 attention mask [B,L]
        uncond_hidden,            # 空字串的 hidden states(CFG 無條件分支)
        uncond_mask,
        num_inference_steps: int = 30,
        early_stop_steps: int = 250,
        lambda_interp: float = 0.15,
        guidance_scale: float = 7.5,
        **kwargs,
    ):
        device = self.device
        if not hasattr(self, "_projector"):
            raise RuntimeError("請先呼叫 set_projector() 掛上訓練好的 MakeupProjector")

        # 1. Encode Source Image x0 -> z0
        source_image = source_image.to(device=device, dtype=self.vae.dtype)
        z0 = self.vae.encode(source_image).latent_dist.sample() * self.vae.config.scaling_factor

        # 2. Early-stopped DDIM Inversion(null-text 分支改走 Projector)
        z_t_star, start_idx = self.invert_llm(
            z0, uncond_hidden, uncond_mask, num_inference_steps, early_stop_steps
        )

        # 3. Reverse Sampling with Interpolation
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps[len(self.scheduler.timesteps) - start_idx - 1:]

        latents = z_t_star

        print(f"[Tier1 Phase 3] Reverse sampling from t={early_stop_steps} "
              f"with lambda={lambda_interp}...")

        for i, t in enumerate(tqdm(timesteps)):
            # 每一步重新產生該時間步專屬的條件向量(timestep-aware)
            uncond_embeds = self._embed(uncond_hidden, uncond_mask, t)
            cond_embeds = self._embed(cond_hidden, cond_mask, t)
            cfg_embeds = torch.cat([uncond_embeds, cond_embeds])

            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=cfg_embeds,
                return_dict=False,
            )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_output = self.scheduler.step(noise_pred, t, latents)
            hat_z0 = step_output.pred_original_sample

            # Interpolation Guidance(與原版相同,顏色錨點機制不變)
            interpolated_z0 = (1 - lambda_interp) * hat_z0 + lambda_interp * masked_image_latents

            prev_t = t - self.scheduler.config.num_train_timesteps // num_inference_steps
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_t]
                if prev_t >= 0
                else torch.tensor(1.0).to(device)
            )
            beta_prod_t_prev = 1 - alpha_prod_t_prev

            latents = (
                alpha_prod_t_prev ** 0.5 * interpolated_z0
                + beta_prod_t_prev ** 0.5 * noise_pred
            )

        # 4. Decode
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])

        return DreamMakeupOutput(images=image, nsfw_content_detected=None)
