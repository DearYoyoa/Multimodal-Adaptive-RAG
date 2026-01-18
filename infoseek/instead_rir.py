import json
import os
import argparse
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

DEVICE = "cuda:3"

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
        return Image.open(image_path)
    else:
        raise ValueError(f"Invalid data_source: {data_source}")

def query_with_image(
        model,
        processor,
        image_path,
        screenshot_path,
        query_text,
        use_screenshot=False,
        exp_dir='experiment/infoseek/',
        sys_msg_filename=None,
        data_source='url'
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

    generated_output = model.generate(**inputs, max_new_tokens=1000, output_hidden_states=True, return_dict_in_generate=True)
    generated_ids = generated_output.sequences
    hidden_states = generated_output.hidden_states
    last_hidden_state = hidden_states[-1][-1]  # 获取最后一个时间步的隐藏状态
    last_hidden_state = last_hidden_state.detach().cpu().numpy()
    np.save(f'data_0921/infoseek/last_hidden_state/hidden_states_{image_path.split("/")[-1].split(".")[0]}.npy', last_hidden_state)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts, messages_record


def main(args):

    processor = AutoProcessor.from_pretrained("experiment/infoseek/models--HuggingFaceM4--idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "experiment/infoseek/models--HuggingFaceM4--idefics2-8b",
    ).to(DEVICE)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('data_0921/infoseek_input.json', 'r') as f:
        data = json.load(f)

    samples = [sample for sample in data]

    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"data_0921/image/{sample['image_id']}.JPEG"
        screenshot_path = f"data_0921/screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response, messages_record = query_with_image(
            model,
            processor,
            image_path,
            screenshot_path,
            query_text,
            use_screenshot=args.use_screenshot,
            exp_dir='experiment/infoseek/',
            sys_msg_filename=args.sys_msg_filename,
            data_source=args.data_source
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

        output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)


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
