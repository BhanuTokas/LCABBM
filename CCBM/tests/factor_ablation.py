# Importing Libraries
import os
import json
import torch
import lpips
from torchvision.transforms import ToTensor
from CCBM.data import get_concept_loaders
from CCBM.ScalingFactorManipulation import ConceptDrifter

# Helper Functions
lpips_loss_fn = lpips.LPIPS(net="alex")


def compute_lpips_distance(img1: torch.tensor, img2: torch.tensor):
    return lpips_loss_fn(img1, img2)


def compute_MSE(img1: torch.tensor, img2: torch.tensor):
    return torch.mean((img1 - img2) ** 2)


# Main function
c = ConceptDrifter(use_ddim=True)
scaling_factors = [0.05 + i * 0.05 for i in range(15)]
scaling_factors.append(c.diff_interface.pipe.vae.config.scaling_factor)
guidance_scales = [
    8 * 2 ** ((i) // 2) + ((i) % 2) * 2 ** ((i + 3) // 2) for i in range(7)
]  # [8, 12, 16, 24, 32, 48, 64]

# Loading Data
NUM_SAMPLES = 25
OUTPUT_DIR = "outputs/factor_ablation_results/"
concept_loaders = get_concept_loaders("broden", None, batch_size=1)
concepts = ["airplane", "greeness", "redness", "bird", "car", "computer", "plant"]
concept_antonyms = {
    "airplane": "car",
    "greeness": "blueness",
    "redness": "blueness",
    "bird": "fish",
    "car": "airplane",
    "computer": "plant",
    "plant": "computer",
}

results = {}
for s_factor in scaling_factors:
    results[s_factor] = {}
    for g_scale in guidance_scales:
        results[s_factor][g_scale] = []
        for concept in concepts:
            text_positive = concept
            text_negative = concept_antonyms.get(concept, "random")
            embeddings = c.clip_interface.getTextEmbedding(
                [text_positive, text_negative]
            )
            z_pos = embeddings[0:1]
            z_neg = embeddings[1:2]
            magnitude = (z_pos.norm(dim=1) + z_neg.norm(dim=1)) / 2
            dataset = concept_loaders[concept]["pos"].dataset
            num_samples = min(NUM_SAMPLES, len(dataset))
            img_set = torch.utils.data.Subset(dataset, list(range(num_samples)))
            for image in img_set:
                print(f"Working on {s_factor=} and {g_scale=}")
                c.diff_interface.pipe.vae.config.scaling_factor = s_factor
                zT, timesteps = c.diff_interface.ddim_invert(
                    image, num_inference_steps=50
                )
                z_i = c.clip_interface.getImgEmbedding(image)
                scale = z_i.norm(dim=1) / magnitude

                # reuse same generator for reconstruction
                gen = torch.Generator(device=c.diff_interface.device).manual_seed(
                    c.diff_interface.seed
                )
                recon = c.diff_interface.reconstruct_from_zT(
                    z_i,
                    zT,
                    num_inference_steps=40,
                    guidance_scale=g_scale,
                    eta=0.0,
                    generator=gen,
                )
                recon_1 = c.diff_interface.reconstruct_from_zT(
                    z_i + (z_pos - z_neg) * scale,
                    zT,
                    num_inference_steps=40,
                    guidance_scale=g_scale,
                    eta=0.0,
                    generator=gen,
                )
                img_tensor = ToTensor()(image).unsqueeze(0)
                recon_tensor = ToTensor()(recon).unsqueeze(0)
                recon_1_tensor = ToTensor()(recon_1).unsqueeze(0)
                lpips_distance_orig = compute_lpips_distance(
                    img_tensor, recon_tensor
                ).item()
                lpips_distance_perturb = compute_lpips_distance(
                    recon_tensor, recon_1_tensor
                ).item()
                mse_distance_orig = compute_MSE(img_tensor, recon_tensor).item()
                mse_distance_perturb = compute_MSE(recon_tensor, recon_1_tensor).item()
                curr_results = {
                    "lpips_distance_orig": lpips_distance_orig,
                    "lpips_distance_perturb": lpips_distance_perturb,
                    "mse_distance_orig": mse_distance_orig,
                    "mse_distance_perturb": mse_distance_perturb
                }
                results[s_factor][g_scale].append(curr_results)

# Saving results
os.makedirs(OUTPUT_DIR, exist_ok=True)
json_path = os.path.join(OUTPUT_DIR, "factor_ablation_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)