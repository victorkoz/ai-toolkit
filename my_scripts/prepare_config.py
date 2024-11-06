import yaml
import os
import asyncio
import aiohttp
# Load the .env file if it exists
from bson.objectid import ObjectId

from my_scripts.connect_mongo import get_client

# Define the YAML structure as a Python dictionary
config_data = {
  "job": "extension",
  "config": {
    "name": "my_first_flux_lora_v1",
    "process": [
      {
        "type": "sd_trainer",
        "training_folder": "output",
        "device": "cuda:0",
        "trigger_word": "",
        "network": {
          "type": "lora",
          "linear": 16,
          "linear_alpha": 16
        },
        "save": {
          "dtype": "float16",
          "save_every": 10,
          "max_step_saves_to_keep": 2,
          "push_to_hub": False
        },
        "datasets": [
          {
            "folder_path": "/path/to/images/folder",
            "caption_ext": "txt",
            "caption_dropout_rate": 0.05,
            "shuffle_tokens": False,
            "cache_latents_to_disk": True,
            "resolution": [
              512,
              768,
              1024
            ]
          }
        ],
        "train": {
          "batch_size": 1,
          "steps": 10,
          "gradient_accumulation_steps": 1,
          "train_unet": True,
          "train_text_encoder": False,
          "gradient_checkpointing": True,
          "noise_scheduler": "flowmatch",
          "optimizer": "adamw8bit",
          "lr": 0.0001,
          "skip_first_sample": True,
          "ema_config": {
            "use_ema": True,
            "ema_decay": 0.99
          },
          "dtype": "bf16"
        },
        "model": {
          "name_or_path": "black-forest-labs/FLUX.1-dev",
          "is_flux": True,
          "quantize": True,
          "low_vram": True
        },
        "sample": {
          "sampler": "flowmatch",
          "sample_every": 10,
          "width": 1024,
          "height": 1024,
          "prompts": [],
          "neg": "",
          "seed": 42,
          "walk_seed": True,
          "guidance_scale": 4,
          "sample_steps": 20
        }
      }
    ]
  },
  "meta": {
    "name": "[name]",
    "version": "1.0"
  }
}

async def prepare_config(taskId):
    try:
      print("Preparing config")
      task = get_client()['tasks'].find_one({"_id": ObjectId(taskId)})

      if task is None:
          raise ValueError(f"Task with ID '{taskId}' not found.")

      dataset_folder_path = f"{os.getenv('ROOT_FOLDER')}/dataset/{taskId}"

      await store_dataset(taskId=taskId, 
                          datasetUrls=task['metadata']['datasetUrls'],
                          dataset_folder_path=dataset_folder_path
                          )
      
      prompt = ''

      if (task['metadata']['gender'] == 'female'):
           prompt = f"A nice picture of {taskId} a woman, close lookup, nice dress, nice gray background"
      else: 
           prompt = f"A nice picture of {taskId} a man, close lookup, nice clothes, nice gray background"
          

      config_data['config']['name'] = taskId
      config_data['config']['process'][0]['trigger_word'] = taskId
      config_data['config']['process'][0]['sample']['prompts'] = [prompt]
      config_data['config']['process'][0]['datasets'][0]['folder_path'] = dataset_folder_path

      folder_path = f"{os.getenv('ROOT_FOLDER')}/config"

      # Determine the path to the config directory
      config_dir = os.path.join(os.path.dirname(__file__), folder_path)
      config_path = os.path.join(config_dir, taskId + '.yml')

      # Create the config directory if it doesn't exist
      os.makedirs(config_dir, exist_ok=True)

      # Write the YAML file
      with open(config_path, "w") as file:
          yaml.dump(config_data, file, default_flow_style=False)

      print(f"Config file saved to {config_path}")
    except Exception as e:
      print(f"Error preparing config: {str(e)}")
      raise
            



async def download_image(session, url, index, folder_path):
    """Download an image and save it to a folder."""
    async with session.get(url) as response:
        if response.status == 200:
            image_data = await response.read()
            image_path = f'{folder_path}/{index}.jpeg'
            with open(image_path, 'wb') as img_file:
                img_file.write(image_data)
            return image_path
        else:
            print(f"Failed to download {url}: {response.status}")

async def create_text_file(index, word, folder_path):
    """Create a text file with a specified word."""
    text_path = f'{folder_path}/{index}.txt'
    with open(text_path, 'w') as txt_file:
        txt_file.write(word)
    return text_path

async def store_dataset(datasetUrls, taskId, dataset_folder_path):
    """Store dataset by downloading images and creating text files."""


    os.makedirs(dataset_folder_path, exist_ok=True)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for index, url in enumerate(datasetUrls):
            # Download image and create text file concurrently
            tasks.append(download_image(session, url, index, dataset_folder_path))
            tasks.append(create_text_file(index, taskId, dataset_folder_path))
        
        await asyncio.gather(*tasks)

    
# if __name__ == '__main__':
#     asyncio.run(prepare_config())
