import json
import os
import argparse
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import torch
DEVICE = "cuda:0"

def download_image(url, save_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as out_file:
            out_file.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download image from {url}")

def load_image_wrapper(image_path, data_source='url'):
    cnt = 0
    if data_source == 'url':
        while cnt < 3:
            try:
                print('attempting to load image...')
                img = load_image(image_path)
                return img
            except Exception as e:
                cnt += 1
                print(f'attempt ({cnt+1}/3) failed with error: {e}')
        raise Exception('Failed to load image')
    elif data_source == 'local':
        return Image.open(image_path).convert('RGB')
    else:
        raise ValueError(f"Invalid data_source: {data_source}")
    
def extract_vision_features(generated_output, inputs, model):

    all_hidden_states = generated_output.hidden_states  # 33 [0-32]

    image_token_id = model.config.image_token_id
    input_ids = inputs['input_ids'][0]  

    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    
    if len(image_token_positions) == 0:
        raise ValueError("No image tokens found in the input sequence")

    gaps = image_token_positions[1:] - image_token_positions[:-1]
    image_boundaries = [0] + ((gaps > 20).nonzero(as_tuple=True)[0] + 1).tolist() + [len(image_token_positions)]

    all_layers_features = []
    for layer_idx, layer_hidden_states in enumerate(all_hidden_states):

        vision_hidden_states = []
        for i in range(len(image_boundaries) - 1):
            start_idx = image_token_positions[image_boundaries[i]]
            end_idx = image_token_positions[image_boundaries[i+1] - 1]

            image_patches = layer_hidden_states[:, start_idx:end_idx+1, :]

            image_feature = image_patches.mean(dim=1)  # [batch_size, hidden_dim]
            vision_hidden_states.append(image_feature)

        layer_features = torch.stack(vision_hidden_states)  # [num_images, batch_size, hidden_dim]
        all_layers_features.append(layer_features)

    all_layers_features = torch.stack(all_layers_features)  # [num_layers, num_images, batch_size, hidden_dim]

    if False:  
        print(f"Found {len(vision_hidden_states)} images")
        print(f"Features shape: {all_layers_features.shape}")
        for i in range(len(image_boundaries) - 1):
            start = image_token_positions[image_boundaries[i]]
            end = image_token_positions[image_boundaries[i+1] - 1]
            num_patches = image_boundaries[i+1] - image_boundaries[i]
            print(f"Image {i}: tokens {start}-{end}, total patches: {num_patches}")
    
    return all_layers_features.detach().cpu().numpy()

def query_with_image(
        model,
        processor,
        image_path,
        screenshot_path,
        query_text,
        use_screenshot=False,
        exp_dir='experiment/evqa/',
        sys_msg_filename=None,
        data_source='url',
        extract_vision_feature=False,
    ):
    with open(exp_dir + sys_msg_filename, 'r') as f:
        query_system_msg = f.read()

    query_image = load_image_wrapper(image_path, data_source=data_source)
    if use_screenshot: 
        screenshot_image = load_image_wrapper(screenshot_path, data_source='local')
        context_text = ("In the screenshot, the large image on the left is the query image for a reverse image search. "
                        "The smaller images on the right and their titles are the top hits from the search. ")
        messages = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image"},
                    {'type': 'text', 'text': context_text},
                    {"type": "image"},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        messages_record = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image", 'url': image_path},
                    {'type': 'text', 'text': context_text},
                    {"type": "image", 'url': screenshot_path},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[query_image, screenshot_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image"},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        messages_record = [
            {
                "role": "user",
                "content": [
                    {'type': 'text', 'text': query_system_msg},
                    {"type": "image", 'url': image_path},
                    {"type": "text", "text": "Query: " + query_text},
                ]
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[query_image], return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # generated_output = model.generate(**inputs, max_new_tokens=1000, output_hidden_states=True, return_dict_in_generate=True)
    with torch.no_grad():
        model_output = model(
            **inputs,
            output_hidden_states=True,  
            return_dict=True,           
        )

    if extract_vision_feature:
        vision_features = extract_vision_features(model_output, inputs, model)
        vision_features = vision_features.squeeze()
        original_vision_features = model_output.image_hidden_states.reshape(2, -1, 4096).mean(dim=1)
        if not os.path.exists(f'hidden_state_i3_train/hidden_evqa/hidden_state/'):
            os.makedirs(f'hidden_state_i3_train/hidden_evqa/hidden_state/')
        np.save(f'hidden_state_i3_train/hidden_evqa/hidden_state/{image_path.split("/")[-1].split(".")[0]}.npy', vision_features)
        if not os.path.exists(f'hidden_state_i3_train/hidden_evqa/original_hidden_state/'):
            os.makedirs(f'hidden_state_i3_train/hidden_evqa/original_hidden_state/')
        np.save(f'hidden_state_i3_train/hidden_evqa/original_hidden_state/{image_path.split("/")[-1].split(".")[0]}.npy', original_vision_features.cpu().numpy())

    generated_ids = 0
    generated_texts = 'For_extract'
    return generated_texts, messages_record


def main(args):

    processor = AutoProcessor.from_pretrained("experiment/infoseek/Idefics3-8B-Llama3")
    model = AutoModelForVision2Seq.from_pretrained(
        "experiment/infoseek/Idefics3-8B-Llama3",
    ).to(DEVICE)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('evqa_train_data/evqa_input.json', 'r') as f:
        data = json.load(f)

    # samples = [_ for cat_data in data for _ in cat_data]
    samples = data
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"evqa_train_data/images/{sample['image_id']}.jpg"
        screenshot_path = f"evqa_train_data/screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response, messages_record = query_with_image(
            model,
            processor,
            image_path,
            screenshot_path,
            query_text,
            use_screenshot=args.use_screenshot,
            exp_dir='experiment/evqa/',
            sys_msg_filename=args.sys_msg_filename,
            data_source=args.data_source,
            extract_vision_feature=True,
        )

        pred = response[0].split('Assistant: ')[-1]
        if isinstance(sample['answer'], list):
            answer_in_pred = any(_.lower() in pred.lower() for _ in sample['answer'])
        else:
            answer_in_pred = sample['answer'].lower() in pred.lower()

        logs.append(
            {
                'idx': idx,
                'answer_in_pred': answer_in_pred,
                'data_id': sample['data_id'],
                'image_id': sample['image_id'],
                'question': sample['question'],
                'answer': sample['answer'],
                'pred': pred,
                'answer_eval': sample['answer_eval'],
                'full_messages_record': str(messages_record),
                'full_response': str(response),
            }
        )

        # output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        # with open(output_name, 'w') as f:
        #     json.dump(logs, f, indent=4)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename', type=str, required=True)
    # screenshot
    argparser.add_argument('--use_screenshot', type=int, required=True)
    # additional
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--data_source', type=str, default='local', choices=['url', 'local'])

    args = argparser.parse_args()
    main(args)
