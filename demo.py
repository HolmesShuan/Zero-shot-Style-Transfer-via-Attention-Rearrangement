import argparse
from zstar.zstar import ReweightCrossAttentionControl
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from diffusers import DDIMScheduler
from zstar.diffuser_utils import ZstarPipeline
from zstar.zstar_utils import AttentionBase
from zstar.zstar_utils import regiter_attention_editor_diffusers
from torchvision.utils import save_image
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from typing import Union
import torch.nn.functional as nnf
import numpy as np
import ptp_utils
import shutil
from torch.optim.adam import Adam
from PIL import Image
import pickle
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MY_TOKEN = ""
TARGET_IMG_SIZE = 560
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
STABLE_DIFFUSION_MODEL_PATH = "./stable-diffusion-v1-5"
SEED = 9999
START_STEP = 5
END_STEP = 30
TOTAL_STEP = 30
NUM_DDIM_STEPS = TOTAL_STEP
LAYER_INDEX = [20, 22, 24, 26, 28, 30]
LAYER_INDEX_STRING = "_".join(str(x) for x in LAYER_INDEX)
print(
    f"Random Seed {SEED}, Style Control [Start Step, End Step] = [{START_STEP}, {END_STEP}], Style Control Layer Index = {LAYER_INDEX}"
)

torch.cuda.set_device(0)  # set the GPU device
seed_everything(SEED)

# Note that you may add your Hugging Face token to get access to the models
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
model = ZstarPipeline.from_pretrained(
    STABLE_DIFFUSION_MODEL_PATH, scheduler=scheduler).to(device)


def load_img_to_numpy(image_path):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    image = np.array(Image.fromarray(image).resize(
        (TARGET_IMG_SIZE, TARGET_IMG_SIZE)))
    return image


def load_image(image_path, device, reverse=False):
    totensor = transforms.ToTensor()
    image = totensor(Image.open(image_path))
    image = image[:3].unsqueeze_(0).float() * 2.0 - 1.0
    image = F.interpolate(image, (TARGET_IMG_SIZE, TARGET_IMG_SIZE))
    if reverse:
        image = torch.flip(image, dims=[2])
    image = image.to(device)
    return image


def get_image(data_dir):
    img_list = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if (
                file.endswith(".jpg")
                or file.endswith(".png")
                or file.endswith(".bmp")
                or file.endswith(".jpeg")
            ):
                img_list.append(os.path.join(root, file))
    assert len(img_list) > 0, "[ERROR] img_list is Empty!"
    return img_list


class NullInversion:

    def prev_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        prev_timestep = (
            timestep
            - self.scheduler.config.num_train_timesteps
            // self.scheduler.num_inference_steps
        )
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.scheduler.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )
        return prev_sample

    def next_step(
        self,
        model_output: Union[torch.FloatTensor, np.ndarray],
        timestep: int,
        sample: Union[torch.FloatTensor, np.ndarray],
    ):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)[
            "sample"
        ]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)[
            "sample"
        ]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_prediction_text - noise_pred_uncond
        )
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.model.text_encoder(
            uncond_input.input_ids.to(self.model.device)
        )[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(
            text_input.input_ids.to(self.model.device)
        )[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        for i in tqdm(range(NUM_DDIM_STEPS)):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(
                    latent_cur, t, cond_embeddings
                )
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(
                    latent_cur, t, uncond_embeddings
                )
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (
                    noise_pred_cond - noise_pred_uncond
                )
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                if loss_item < epsilon + i * 2e-5:
                    break
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        return uncond_embeddings_list

    def invert(
        self,
        image_path: str,
        prompt: str,
        num_inner_steps=10,
        early_stop_epsilon=1e-5,
        verbose=False
    ):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_img_to_numpy(image_path)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)
        if verbose:
            print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(
            ddim_latents, num_inner_steps, early_stop_epsilon
        )
        return (image_gt, image_rec), ddim_latents, ddim_latents[-1], uncond_embeddings

    def __init__(self, model):
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None


def parse_args():
    parser = argparse.ArgumentParser(description="settings")
    parser.add_argument('--sub_exp_name', type=str,
                        default="workdir/demo", help='sub exp name')
    parser.add_argument('--content_img_folder', type=str,
                        default="./content_images/", help='content image paths')
    parser.add_argument('--style_img_folder', type=str,
                        default="./style_images/", help='style image paths')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    SUB_EXP_NAME = args.sub_exp_name
    CONTENT_IMG_FOLDER = args.content_img_folder
    STYLE_IMG_FOLDER = args.style_img_folder

    null_inversion = NullInversion(model)

    if os.path.exists(SUB_EXP_NAME):
        shutil.rmtree(SUB_EXP_NAME)
    os.makedirs(SUB_EXP_NAME, exist_ok=True)

    content_list = get_image(CONTENT_IMG_FOLDER)
    style_list = get_image(STYLE_IMG_FOLDER)
    for content in content_list:
        content_image = load_image(content, device)
        source_prompt = ""
        target_prompt = ""
        prompts = [source_prompt, target_prompt]
        pickle_file_name = (
            content.replace(".png", ".pkl")
            .replace(".jpg", ".pkl")
            .replace(".jpeg", ".pkl")
            .replace(".bmp", ".pkl")
        )
        if os.path.isfile(pickle_file_name):
            with open(pickle_file_name, "rb") as f:
                pre_computed_data = pickle.load(f)
            content_latent_list = pre_computed_data[0]
            x_t = pre_computed_data[1]
            uncond_embeddings = pre_computed_data[2]
        else:
            _, content_latent_list, x_t, uncond_embeddings = null_inversion.invert(
                content, prompts, verbose=True
            )
            with open(pickle_file_name, "wb") as f:
                pickle.dump([content_latent_list, x_t, uncond_embeddings], f)
        start_code_content = x_t.expand(len(prompts), -1, -1, -1)
        for style in style_list:
            style_image = load_image(style, device)
            editor = AttentionBase()
            regiter_attention_editor_diffusers(model, editor)
            _, style_latent_list = model.invert(
                style_image,
                source_prompt,
                guidance_scale=7.5,
                num_inference_steps=TOTAL_STEP,
                return_intermediates=True,
            )
            # hijack the attention module
            editor = ReweightCrossAttentionControl(
                START_STEP,
                END_STEP,
                layer_idx=LAYER_INDEX,
                total_steps=TOTAL_STEP,
                content_img_name=content,
            )
            regiter_attention_editor_diffusers(model, editor)
            # inference the synthesized image
            image_stylized = model(
                prompts,
                latents=start_code_content,
                guidance_scale=7.5,
                uncond_embeddings=uncond_embeddings,
                num_inference_steps=TOTAL_STEP,
                ref_intermediate_latents=[
                    content_latent_list, style_latent_list],
            )
            # Note: querying the inversion intermediate features latents_list
            # may obtain better reconstruction and editing results
            full_image_path = os.path.join(
                SUB_EXP_NAME,
                content.split("/")[-1][:-4] + "_" +
                style.split("/")[-1][:-4] + ".png",
            )
            # save the synthesized image
            out_image = torch.cat(
                [
                    content_image * 0.5 + 0.5,
                    image_stylized[0:1],
                    image_stylized[-1:],
                ],
                dim=0,
            )
            save_image(out_image, full_image_path)
            full_image_name = os.path.basename(full_image_path)
            full_image_name = os.path.splitext(full_image_name)[0]
            save_image(
                content_image * 0.5 + 0.5,
                os.path.join(SUB_EXP_NAME, full_image_name + "_source.png"),
            )
            save_image(
                image_stylized[0:1], os.path.join(
                    SUB_EXP_NAME, full_image_name + "_style.png")
            )
            save_image(
                image_stylized[-1:],
                os.path.join(SUB_EXP_NAME, full_image_name +
                             "_reconstructed.png"),
            )
            print("Syntheiszed images are saved in ", full_image_path)


if __name__ == "__main__":
    main()
