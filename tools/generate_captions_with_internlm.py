import os
# Set the HF_ENDPOINT environment variable
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # before import all hf related pkgs

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import tyro
from tqdm.auto import tqdm
import random
from PIL import Image

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image


torch.backends.cudnn.benchmark = True

torch.set_grad_enabled(False)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

querys = [
    "<|User|> <ImageHere> This is a view of a scene. Please describe this scene in detail.",
    "<|User|> Please compress the description into three or four sentences. The details about the visual contents, features, object relationships and background information must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object.\
    Remove all words like 'The image' or 'The scene' in your description.",
    # "<|User|> Please further compress the description into two or three sentences. The details about the main visual contents and apperance features must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object.\
    # Remove all words like 'The image' or 'The scene' in your description.",
    # "<|User|> Please further compress the description into one sentence. The main visual contents must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object. You should use different words if possible, campared to your previous responses.\
    # Remove all words like 'The image' or 'The scene' in your description.",
    "<|User|> Please further compress the description into one sentence. The main visual contents must be retained. Please remove the irrelevant and unnecessary comments, contents that are not related to the object. You should use different words if possible, campared to your previous responses.\
    Remove all words like 'The image' or 'The scene' in your description.",
]

@torch.no_grad()
def generate_internlm_captions(
    # path: str = "/nas6/yyb/MVImgNet",
    path: str = "/nas6/yyb/DL3DV-10K", 
    model_type: str = 'internlmx',
    use_4bit: bool = True,
    category: str = 'all',
    hf_path: str = "/data0/hf_weights",
    device: str = "cuda:0",
    num_views: int = 1,
    begin: float = 0,
    end: float = 1,
):    
    # init model and tokenizer
    
    
    if model_type == 'internlmx':
        if use_4bit:
            
            from transformers import AutoModel
            import auto_gptq
            from auto_gptq.modeling import BaseGPTQForCausalLM
            from transformers import AutoModelForCausalLM, AutoTokenizer
            auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
            class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
                layers_block_name = "model.layers"
                outside_layer_modules = [
                    'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
                ]
                inside_layer_modules = [
                    ["attention.wqkv.linear"],
                    ["attention.wo.linear"],
                    ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                    ["feed_forward.w2.linear"],
                ]
 
            model_path = os.path.join(hf_path, 'internlm/internlm-xcomposer2-vl-7b-4bit')
            model = InternLMXComposer2QForCausalLM.from_quantized(
            model_path, trust_remote_code=True,use_marlin=True).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)
        else:
            model_path = os.path.join(hf_path, 'internlm/internlm-xcomposer2-vl-7b')
            model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == 'mobilevlm':
        print('Use mtgv/MobileVLM_V2-1.7B')
        model_path = os.path.join(hf_path, 'mtgv/MobileVLM_V2-1.7B')
        model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif model_type == 'internvl':
        # Load model directly
        from transformers import AutoModel
        from transformers import AutoTokenizer, AutoModel
        model_path = os.path.join(hf_path, "OpenGVLab/InternVL2-1B")
        #model_path = "OpenGVLab/InternVL2-1B"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    basedirs = []   
    if "dl3dv" in  path.lower(): #prasing dl3dv
        img_folder_name = 'images_4'
        for scene_id in sorted(os.listdir(os.path.join(path))):
            if not os.path.isdir(os.path.join(path, scene_id)):
                continue
            # if os.path.exists(os.path.join(path, scene_id, 'captions.txt')):
            #     continue
            if not os.path.exists(os.path.join(path, scene_id, img_folder_name)):
                continue
            basedirs.append(os.path.join(path, scene_id))
    elif 'mvimgnet' in path.lower(): #prasing mvimagenet 
        img_folder_name = 'images'
        if category == 'all':
            categories = os.listdir(path)
        else:
            categories = [category]
        basedirs = []
        
        for category in categories:
            for id in os.listdir(os.path.join(path, category)):
                if not os.path.isdir(os.path.join(path, category, id, img_folder_name)):
                    continue
                basedirs.append(os.path.join(path, category, id))
        random.shuffle(basedirs)
    else:
        raise ValueError
    basedirs = basedirs[int(begin * len(basedirs)):int(end * len(basedirs))]

    # loop over data
    for idx, basedir in enumerate(tqdm(basedirs, desc="Generate Captions")):
        # get sequence, category from batch
        filenames = [f for f in sorted(os.listdir(os.path.join(basedir, img_folder_name))) if (f.endswith('jpg')or f.endswith('png'))]

        # try:
        if True:
            if os.path.exists(os.path.join(basedir, 'captions.txt')):
                print('skip')
                continue
            if len(filenames) < 1: 
                with open(os.path.join(basedir, 'captions.txt'), 'w') as f:
                    pass
                continue

            if len(filenames) < num_views: 
                filenames = random.choices(filenames, k=num_views)
            else:
                filenames = random.sample(filenames, k=num_views)

            # get image from batch
            images = [os.path.join(basedir, img_folder_name, filename) for filename in filenames]
            results = []

            # run captioning
            for image in images:

                history = ''

                for i, query in enumerate(querys):
                    history += query
                    if model_type == 'internlmx':
                        with torch.cuda.amp.autocast():
                            response, _ = model.chat(tokenizer, 
                                                        query=history, 
                                                        image=image, 
                                                        history=[], 
                                                        do_sample=False)
                    else:
                            pixel_values = load_image(image, max_num=12).to(torch.bfloat16).to(device)
                            generation_config = dict(max_new_tokens=1024, do_sample=False)
                            response, _ = model.chat(tokenizer, 
                                                    pixel_values, 
                                                    history, 
                                                    generation_config,
                                                    history=None, 
                                                    return_history=True
                                                    )
                    history += " <|Bot|>" + response
                    if i > 0:
                        results += [response]

            # save captions
            if True:
                print(results)
            with open(os.path.join(basedir, 'captions.txt'), 'w', encoding='utf-8') as f:
                f.write('\n'.join(results))
        # except Exception as e:
        #     print(f'{Exception}')
        #     with open(os.path.join(basedir, 'captions.txt'), 'w') as f:
        #         print(f"Skip {basedir}.")
        #         pass

if __name__ == "__main__":
    tyro.cli(generate_internlm_captions)
