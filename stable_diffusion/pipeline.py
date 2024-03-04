import torch 
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, do_cfg=True, 
             cfg_scale=7.5, sampler_name="ddpm", n_inference_steps=50, models=(), seed=None,
             device=None, idle_device=None, tokenizer=None):
    
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models['clip']
        clip.to(device)

        if do_cfg:
            # convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids

            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)

            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)

            uncond_context = clip(uncond_tokens)

            #(2, seq_len, dim) = (2,77,768)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            
            #(1, seq_len, dim) = (1,77,768)
            context = clip(tokens)


        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError(f"Unknown sampler {sampler_name}")
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)

            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)

            input_image_tensor = rescale(input_image_tensor, (0,255), (-1,1))

            # Add batch dimension. (height, width, channel) -> (batch_size, height, width, channel)
            input_image_tensor = input_image_tensor.unsequeeze(0)

            # (batch_size, height, width, channel) -> (batch_size, channel, height, width)
            input_image_tensor = input_image_tensor.permute(0,3,1,2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # run the iamge through the encoder of the VAE 

            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strenght(strength=strength)

            latents = sampler.add_noise(latents, sampler.timestep(0))

            to_idle(encoder)

        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models['diffusion']

        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)

            time_embedding = get_time_embedding(timestep).to(device)

            # (batch_size, 4, latents_height, latents_width)
            model_input = latents

            if do_cfg:
                # (batch_size, 4, latents_height, latents_width) -> (2 * batch_size, latents_height, latents_width)
                model_input = model_input.repeat(2,1,1,1)

            # model_output is the predicted noise by the UNET
            model_output = diffusion(model_input, )




