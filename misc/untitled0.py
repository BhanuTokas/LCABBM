import torch
from tqdm import tqdm
from PIL import Image
from typing import Union
import numpy as np
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from diffusers import DiffusionPipeline


def enable_memory_savings(pipe):
    # attempt common memory reductions
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()
    except Exception:
        pass
    return pipe


def build_fixed_latent(
    pipe, height, width, device, latent_scale=8, seed=0, dtype=torch.float32
):
    """
    Build a fixed random latent tensor to pass to decoder.
    Assumes latent spatial dims = height/latent_scale, width/latent_scale.
    Returns: (latents, generator)
    """
    b = 1
    # infer channels from UNet if available
    unet = (
        getattr(pipe, "decoder", None)
        or getattr(pipe, "unet", None)
        or getattr(pipe, "unet", None)
    )
    if unet is None:
        raise RuntimeError(
            "Pipeline does not expose an accessible UNet (decoder/unet)."
        )
    in_ch = getattr(unet, "in_channels", None)
    if in_ch is None:
        # fallback channel for stable diffusion style latents
        in_ch = 4
    latent_h = height // latent_scale
    latent_w = width // latent_scale
    torch_generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (b, in_ch, latent_h, latent_w),
        generator=torch_generator,
        device=device,
        dtype=dtype,
    )
    return latents, torch_generator


class CLIPInterface:

    def __init__(
        self,
        clip_model="openai/clip-vit-large-patch14",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.dtype = dtype
        self.device = device
        self.clip_model = clip_model
        self.initCLIP()

    def initCLIP(self):
        self.clip_model_default = self.clip_model
        self.clip_processor = CLIPImageProcessor.from_pretrained(self.clip_model)
        self.clip_image_encoder = (
            CLIPVisionModelWithProjection.from_pretrained(self.clip_model)
            .to(self.device)
            .eval()
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(self.clip_model)
        self.text_encoder = (
            CLIPTextModelWithProjection.from_pretrained(self.clip_model)
            .to(self.device)
            .eval()
        )

    def getImgEmbedding(self, img: Image):
        # CLIP processor is usually callable; wrap defensively
        clip_inputs = self.clip_processor(images=img, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            clip_out = self.clip_image_encoder(**clip_inputs)
            if hasattr(clip_out, "image_embeds"):
                z_i = clip_out.image_embeds
            else:
                # fallback pool
                z_i = clip_out.last_hidden_state.mean(dim=1)
        return z_i

    def getTextEmbedding(self, text: Union[list[str], str]):
        toks = self.tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            out = self.text_encoder(**toks)
            if hasattr(out, "text_embeds"):
                return out.text_embeds
            else:
                return out.last_hidden_state.mean(dim=1)


class DiffusionInterface:

    def __init__(
        self,
        model_id="diffusers/stable-diffusion-2-1-unclip-i2i-l",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        height=512,
        width=512,
        latent_scale=8,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
    ):
        """
        use_ddim: if True, replace the pipeline scheduler with DDIMScheduler
        ddim_eta: the eta parameter used by DDIM (0.0 = deterministic)
        """
        self.dtype = dtype
        self.seed = seed
        self.height = height
        self.width = width
        self.latent_scale = latent_scale
        self.device = device
        self.model_id = model_id
        self.use_ddim = use_ddim
        self.ddim_eta = ddim_eta
        self.generator = None
        self.initDiffusionPipeline()
        self.getLatents()

    def getLatents(self):
        self.latents, self.generator = build_fixed_latent(
            self.pipe,
            self.height,
            self.width,
            self.device,
            latent_scale=self.latent_scale,
            seed=self.seed,
            dtype=self.dtype,
        )

    def initDiffusionPipeline(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=self.dtype, trust_remote_code=True
        )
        # optionally switch to DDIM scheduler
        if self.use_ddim:
            try:
                # import locally to avoid requiring it when not used
                from diffusers import DDIMScheduler

                # construct a DDIM scheduler using the existing scheduler's config
                self.pipe.scheduler = DDIMScheduler.from_config(
                    self.pipe.scheduler.config
                )
            except Exception as e:
                # if DDIM isn't available or replacement fails, raise informative error
                raise RuntimeError(f"Failed to set DDIM scheduler: {e}")
        self.pipe = enable_memory_savings(self.pipe)
        self.pipe = self.pipe.to(self.device)

    def genImage(self, z, num_inference_steps=40, guidance_scale=8.0):
        """
        Generate image(s) from CLIP embeddings (z) and fixed latents.
        If use_ddim=True the eta parameter will be passed to the pipeline call.
        """
        z = z.type(self.dtype).to(self.device)
        # ensure latents dtype/device
        self.latents = self.latents.to(device=self.device, dtype=self.dtype)
        # Build kwargs for the pipeline call
        call_kwargs = dict(
            image_embeds=z,
            latents=self.latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=self.generator,
        )
        if self.use_ddim:
            # DDIM supports 'eta' (controls stochasticity)
            call_kwargs["eta"] = float(self.ddim_eta)

        out = self.pipe(**call_kwargs)
        images = out.images if hasattr(out, "images") else out
        img = images[0] if isinstance(images, (list, tuple)) else images
        return img

    # ----- New helpers for encoding/decoding and DDIM inversion -----
    def pil_to_tensor(self, pil_image: Image, height: int, width: int, device, dtype):
        """Convert PIL image -> normalized tensor (1,3,H,W) suitable for VAE.encode."""
        pil_rgb = pil_image.convert("RGB").resize(
            (width, height), resample=Image.BICUBIC
        )
        arr = np.asarray(pil_rgb).astype(np.float32) / 255.0  # [0,1]
        # common diffusers VAE expects [-1,1]
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(device=device, dtype=dtype)

    def encode_image_to_latents(self, pil_image: Image):
        """
        Robust encode -> latent using pipeline VAE without assuming image_processor is callable.
        Returns latents shaped (1,C_lat,H_lat,W_lat).
        """
        pipe = self.pipe
        device = self.device
        dtype = self.dtype

        # 1) Prefer pipeline's VaeImageProcessor.preprocess()
        proc = getattr(pipe, "image_processor", None)
        img_tensor = None
        if proc is not None and hasattr(proc, "preprocess"):
            try:
                result = proc.preprocess(
                    pil_image, height=self.height, width=self.width
                )
                if isinstance(result, torch.Tensor):
                    img_tensor = result.to(device=device, dtype=dtype)
            except Exception:
                img_tensor = None

        # 2) fallback manual conversion
        if img_tensor is None:
            img_tensor = self.pil_to_tensor(
                pil_image, self.height, self.width, device, dtype
            )

        # 3) encode with VAE — use mode() (mean) not sample() for deterministic inversion
        with torch.no_grad():
            enc_out = pipe.vae.encode(img_tensor)
            if hasattr(enc_out, "latent_dist"):
                latents = enc_out.latent_dist.mode()
            elif isinstance(enc_out, (tuple, list)):
                latents = enc_out[0]
            else:
                latents = enc_out

        # 4) Apply scaling factor — the pipeline's denoising loop operates on scaled latents
        # (confirmed by __call__ which decodes with `latents / vae.config.scaling_factor`)
        latents = latents * pipe.vae.config.scaling_factor
        return latents

    def decode_latents_to_image(self, latents):
        """Decode VAE latents back to a PIL image using pipeline VAE.decode."""
        pipe = self.pipe
        device = self.device
        dtype = self.dtype
        lat = latents.to(device=device, dtype=dtype)
        with torch.no_grad():
            dec = pipe.vae.decode(lat)
            # handle ModelOutput or tensor
            if hasattr(dec, "sample"):
                imgs = dec.sample
            elif isinstance(dec, (tuple, list)):
                imgs = dec[0]
            else:
                imgs = dec
        # expected [-1,1] -> [0,1]
        imgs = (imgs.clamp(-1, 1) + 1) / 2
        imgs = (imgs * 255).round().to(torch.uint8).cpu().numpy()
        # (B,C,H,W) -> HWC
        if imgs.ndim == 4 and imgs.shape[1] == 3:
            arr = imgs[0].transpose(1, 2, 0)
        else:
            arr = imgs[0]
        return Image.fromarray(arr)

    def ddim_invert(self, pil_image: Image, z_I, num_inference_steps=50):
        """
        Find the initial latent z_T such that the pipeline's denoising loop starting from
        z_T, conditioned on z_I, reproduces pil_image. This is true DDIM inversion.

        Returns: z_T (tensor), timesteps (descending, for use in reconstruct_from_zT)
        """
        pipe = self.pipe
        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps  # descending [T, ..., 0]

        # Encode image to scaled VAE latent x0.
        # The pipeline decodes with `latents / scaling_factor`, so the loop operates
        # in scaled latent space — x0 must be scaled to match.
        x0 = self.encode_image_to_latents(pil_image)  # already scaled by scaling_factor
        x_t = x0.clone().to(self.device, dtype=self.dtype)

        # Build class_labels exactly as __call__ does:
        # pipe.noise_image_embeddings(z_I, noise_level=0) ->
        #   scale -> add_noise(noise_level=0) -> unscale -> cat(get_timestep_embedding)
        # giving shape (1, 1536) with no CFG duplication (guidance_scale=1 for inversion).
        with torch.no_grad():
            cls_cond = pipe.noise_image_embeddings(
                image_embeds=z_I.to(device=self.device, dtype=self.dtype),
                noise_level=0,
                generator=None,
            )

        # Build prompt_embeds (empty string, no CFG) — the UNet needs encoder_hidden_states.
        with torch.no_grad():
            prompt_embeds, _ = pipe.encode_prompt(
                prompt="",
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

        # Ascending timesteps: [t0, ..., T]
        inversion_timesteps = list(reversed(timesteps.tolist()))

        for i in tqdm(range(len(inversion_timesteps) - 1)):
            t = inversion_timesteps[i]
            t_next = inversion_timesteps[i + 1]
            t_tensor = torch.tensor(t, device=self.device, dtype=torch.long)

            latent_input = scheduler.scale_model_input(x_t, t_tensor)

            with torch.no_grad():
                noise_pred = pipe.unet(
                    latent_input,
                    t_tensor,
                    encoder_hidden_states=prompt_embeds,
                    class_labels=cls_cond,
                    return_dict=False,
                )[0]

            alpha_t = scheduler.alphas_cumprod[t].to(self.device, dtype=self.dtype)
            alpha_t_next = scheduler.alphas_cumprod[t_next].to(
                self.device, dtype=self.dtype
            )

            # Inverse DDIM step: x_{t+1} from x_t and predicted noise
            pred_x0 = (x_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
            x_t = alpha_t_next.sqrt() * pred_x0 + (1 - alpha_t_next).sqrt() * noise_pred

        # The pipeline's prepare_latents multiplies randn by scheduler.init_noise_sigma.
        # Since we pass z_T directly as `latents` (bypassing prepare_latents), we must
        # pre-divide by init_noise_sigma so the pipeline's internal scaling is consistent.
        z_T = x_t / scheduler.init_noise_sigma
        return z_T, timesteps

    def reconstruct_from_zT(
        self,
        z_I,
        z_T,
        num_inference_steps=50,
        guidance_scale=1.0,
        eta=0.0,
        generator=None,
    ):
        """
        Reconstruct (or concept-perturb) an image from:
          z_I : CLIP image embedding (original for reconstruction, perturbed to shift concept)
          z_T : the inverted latent noise from ddim_invert — encodes the spatial structure

        Passes z_T as `latents` to seed the denoising loop with the image-specific noise
        found by inversion, so the output is structurally grounded in the original image.
        """
        device = self.device
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        out = self.pipe(
            prompt="",
            image_embeds=z_I.to(self.dtype),
            latents=z_T.to(device=device, dtype=self.dtype),
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            noise_level=0,
            eta=float(eta) if self.use_ddim else None,
        )
        images = out.images if hasattr(out, "images") else out
        img = images[0] if isinstance(images, (list, tuple)) else images
        if isinstance(img, Image.Image):
            return img
        if isinstance(img, torch.Tensor):
            img_tensor = img.detach().cpu()
            if img_tensor.ndim == 4:
                img_tensor = img_tensor[0]
            img_tensor = (img_tensor.clamp(-1, 1) + 1) / 2
            arr = (img_tensor * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr)
        import numpy as _np

        if isinstance(img, _np.ndarray):
            return Image.fromarray(img)
        raise RuntimeError(f"Unhandled pipeline output type: {type(img)}")


class ConceptDrifter:

    def __init__(
        self,
        model_id="diffusers/stable-diffusion-2-1-unclip-i2i-l",
        clip_model="openai/clip-vit-large-patch14",
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=42,
        height=512,
        width=512,
        latent_scale=8,
        use_ddim: bool = False,
        ddim_eta: float = 0.0,
    ):
        self.device = device
        self.clip_interface = CLIPInterface(clip_model, dtype, device)
        self.diff_interface = DiffusionInterface(
            model_id,
            dtype,
            device,
            seed,
            height,
            width,
            latent_scale,
            use_ddim,
            ddim_eta,
        )

    def getConceptVector(self, positive_text: str, negative_text: str):
        embeddings = self.clip_interface.getTextEmbedding(
            [positive_text, negative_text]
        )
        pos = embeddings[0:1]
        neg = embeddings[1:2]
        magnitude = (pos.norm(dim=1) + neg.norm(dim=1)) / 2
        return (pos - neg), magnitude

    def getProbs(self, z_i, z_pos, z_neg):
        prob_p = torch.nn.functional.cosine_similarity(z_i, z_pos)
        prob_n = torch.nn.functional.cosine_similarity(z_i, z_neg)
        probs = torch.nn.functional.softmax(torch.tensor([prob_p, prob_n]))
        return probs

    def perturbImagePoints(self, img, z_pos, z_neg, delta=0.1):
        z_i = self.clip_interface.getImgEmbedding(img)
        z_i_mag = z_i.norm(dim=1)
        vector = z_pos - z_neg
        magnitude = (z_pos.norm(dim=1) + z_neg.norm(dim=1)) / 2
        probs = self.getProbs(z_i, z_pos, z_neg)
        if probs[0] > probs[1]:
            vector = -1 * vector
        z_new = z_i + (delta * vector * (z_i_mag / magnitude))
        z_i = z_i.to(self.device)
        z_new = z_new.to(self.device)
        # Invert with z_i to find the latent that reproduces this image
        z_T, _ = self.diff_interface.ddim_invert(img, z_i)
        orig_img = self.diff_interface.reconstruct_from_zT(z_i, z_T)
        new_img = self.diff_interface.reconstruct_from_zT(z_new, z_T)
        if probs[0] > probs[1]:
            return new_img, orig_img
        else:
            return orig_img, new_img

    """        
    def perturbImageVector(self, img, vector, magnitude, delta=0.1):
        # More efficient to use if same concept needs to be applied to multiple images.
        z_i = self.clip_interface.getImgEmbedding(img)
        z_i_mag = z_i.norm(dim=1)
        z_new = z_i + (delta * vector * (z_i_mag / magnitude))
        orig_img = self.diff_interface.genImage(z_i)
        new_img = self.diff_interface.genImage(z_new)
        return orig_img, new_img

    def applyConcept(self, img, positive_text, negative_text, delta=0.1):
        # Easier to use for user.
        vector, mag = self.getConceptVector(positive_text, negative_text)
        orig_img, new_img = self.perturbImageVector(img, vector, mag, delta)
        return orig_img, new_img
    """


if __name__ == "__main__":
    # Example: enable DDIM with deterministic sampling (eta=0.0)
    c = ConceptDrifter(use_ddim=True)
    image = Image.open("samples/test_img.jpg").resize((512, 512))
    text_positive = "yellow"
    text_negative = "blue"
    embeddings = c.clip_interface.getTextEmbedding([text_positive, text_negative])
    z_pos = embeddings[0:1]
    z_neg = embeddings[1:2]
    # reuse same generator for reconstruction
    gen = torch.Generator(device=c.diff_interface.device).manual_seed(
        c.diff_interface.seed
    )

    # Example: invert and reconstruct deterministically
    z_i = c.clip_interface.getImgEmbedding(image)
    zT, timesteps = c.diff_interface.ddim_invert(image, z_i, num_inference_steps=50)
    recon = c.diff_interface.reconstruct_from_zT(
        z_i, zT, num_inference_steps=50, guidance_scale=1.0, eta=0.0, generator=gen
    )
    recon.save("reconstructed_from_zT.png")

    img1, img2 = c.perturbImagePoints(image, z_pos, z_neg, delta=0.2)
