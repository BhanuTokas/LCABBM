import torch
from tqdm import tqdm
from PIL import Image
from typing import Union
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
                self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
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
    
    def ddim_invert(self, pil_image: Image, num_inference_steps=50):
        """
        Perform DDIM inversion: image -> latents (via VAE) -> invert the deterministic DDIM sampling
        to obtain the initial noise latent z_T that, when sampled with the same scheduler & steps,
        reproduces the input.
        Returns: z_T (tensor), and the timesteps used (torch.Tensor)
        """
        # 1) Encode to latents (x_0 in latent space)
        x0 = self.encode_image_to_latents(pil_image)  # shape (1,C,H,W)
        # Ensure scheduler is DDIM-compatible
        scheduler = self.pipe.scheduler
        # set timesteps for inversion consistent with sampling
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps  # descending timesteps
        # convert x0 to the noisy latents corresponding to each timestep by running predictive step
        # We will compute z_t iteratively forward from x0 -> z_T using DDIM forward equations (deterministic).
        # Implementation reference: DDIM paper eqns. We invert the process by computing x_t from x_{t-1} (forward direction).
        x_t = x0.clone().to(self.device, dtype=self.dtype)
        for i, t in enumerate(tqdm(list(timesteps))):
            # For DDIM forward update we need alpha cumprod info on scheduler
            alpha_prod_t = scheduler.alphas_cumprod[t]
            alpha_prod_prev = scheduler.alphas_cumprod[t] if i == len(timesteps)-1 else scheduler.alphas_cumprod[timesteps[i+1]]
            # Convert scalars to tensors
            a_t = torch.tensor(alpha_prod_t, device=self.device, dtype=self.dtype)
            a_prev = torch.tensor(alpha_prod_prev, device=self.device, dtype=self.dtype)
            # Predict epsilon from x_t using the denoiser (UNet) — note: in forward direction we treat x_t as current "clean" and compute x_{t+1}
            # For forward DDIM step we need the noise epsilon_t that would produce x_t from x0. However computing a robust forward mapping is subtle.
            # Simpler approach: use scheduler.add_noise to simulate adding noise at a given timestep from a fixed random noise.
            # We'll create a deterministic pseudo-random noise for each step using generator seeded by self.seed
            noise = torch.randn_like(x_t, generator=torch.Generator(device=self.device).manual_seed(self.seed))
            x_t = scheduler.add_noise(x0, noise, t)
        z_T = x_t
        return z_T, timesteps

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
            model_id, dtype, device, seed, height, width, latent_scale, use_ddim, ddim_eta
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
        vector = z_pos-z_neg
        magnitude = (z_pos.norm(dim=1) + z_neg.norm(dim=1)) / 2
        probs = self.getProbs(z_i, z_pos, z_neg)            
        if(probs[0]>probs[1]):
            vector = -1 * vector
        z_new = z_i + (delta * vector * (z_i_mag / magnitude))
        z_i = z_i.to(self.device)
        z_new = z_new.to(self.device)
        orig_img = self.diff_interface.genImage(z_i)
        new_img = self.diff_interface.genImage(z_new)
        if(probs[0]>probs[1]):
            return new_img, orig_img
        else:
            return orig_img, new_img
    '''        
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
    '''


if __name__ == "__main__":
    # Example: enable DDIM with deterministic sampling (eta=0.0)
    c = ConceptDrifter(use_ddim=True, ddim_eta=0.0)
    image = Image.open("samples/test_img2.jpg")
    text_positive = "yellow"
    text_negative = "blue"
    embeddings = c.clip_interface.getTextEmbedding(
        [text_positive, text_negative]
    )
    z_pos = embeddings[0:1]
    z_neg = embeddings[1:2]
    img1, img2 = c.perturbImagePoints(image, z_pos, z_neg, delta=0.2)