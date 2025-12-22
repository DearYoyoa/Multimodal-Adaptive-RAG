import json
import os
import argparse
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
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


def load_image_wrapper(image_path, data_source='local'):
    cnt = 0
    if data_source == 'url':
        while cnt < 3:
            try:
                print('attempting to load image...')
                img = load_image(image_path)
                return img
            except Exception as e:
                cnt += 1
                print(f'attempt ({cnt + 1}/3) failed with error: {e}')
        raise Exception('Failed to load image')
    elif data_source == 'local':
        return Image.open(image_path)
    else:
        raise ValueError(f"Invalid data_source: {data_source}")

def parse_response(text):
    def extract(tag):
        if tag not in text:
            return ""
        return text.split(tag)[1].split("\n[")[0].strip()

    reasoning = extract("[REASONING]")
    decision = extract("[DECISION]")
    final_answer = extract("[FINAL_ANSWER]")
    return reasoning, decision, final_answer

def cot_inference(
        model,
        processor,
        query_image,
        screenshot_image,
        query_text,
        query_system_msg=None,
):
    with open(query_system_msg, 'r') as f:
        query_system_msg = f.read()
    query_image = load_image_wrapper(query_image, data_source='local')
    screenshot_image = load_image_wrapper(screenshot_image, data_source='local')
    final_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query_system_msg},
                {"type": "text", "text": "Image 1: Original image."},
                {"type": "image"},

                {"type": "text", "text": "Image 2: Screenshot of Google reverse image search results."},
                {"type": "image"},
                {"type": "text", "text": "Question: " + query_text},
            ]
        }
    ]
    images = [query_image, screenshot_image]
    prompt = processor.apply_chat_template(final_messages, add_generation_prompt=True)
    # print("prompt:", prompt)
    inputs = processor(text=prompt, images=images, return_tensors="pt").to(DEVICE)
    generated = model.generate(**inputs, max_new_tokens=128)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = generated[0][input_len:]
    answer = processor.decode(gen_ids, skip_special_tokens=True)
    # print("answer", answer)
    return answer, final_messages

def main(args):

    processor = AutoProcessor.from_pretrained("model/qwen7b")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("model/qwen7b",
                                                            ignore_mismatched_sizes=True, device_map="cuda:0",
                                                            torch_dtype=torch.float16)
    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)
    data_path = args.data_path
    # load sample data
    with open(f'{data_path}/input.json', 'r') as f:
        data = json.load(f)

    samples = [sample for sample in data]

    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)
    skip_cnt = 0
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"{data_path}/image/{sample['image_id']}.jpg"
        screenshot_path = f"{data_path}/screenshot/{sample['image_id']}-search.png"
        # image_path = f"{data_path}/image/{sample['image_id']}"
        # screenshot_path = f"{data_path}/screenshot/{sample['image_id'].split('.')[0]}-search.png"
        if not os.path.exists(image_path):
            print(f"[Skip] image not found: {image_path}")
            skip_cnt += 1
            continue

        if not os.path.exists(screenshot_path):
            print(f"[Skip] screenshot not found: {screenshot_path}")
            skip_cnt += 1
            continue
        query_text = sample['question']
        response, final_messages = cot_inference(
            model,
            processor,
            image_path,
            screenshot_path,
            query_text,
            query_system_msg=args.sys_msg_filename,
        )

        # pred = response.split('assistant\\n')[-1]
        # pred = response
        reasoning, decision, pred = parse_response(response)
        # print("response", response)
        # print("decision", decision)
        # print("pred", pred)
        pred = response
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
                "decision": decision,
                "reasoning": reasoning,
                'pred': pred,
                'answer_eval': sample['answer_eval'],
                'full_response': str(response),
            }
        )

        output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)
    print(f"Finished. Skipped {skip_cnt} samples.")


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--data_path', type=str, required=True)
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename', type=str, required=True)
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--data_source', type=str, default='local', choices=['url', 'local'])

    args = argparser.parse_args()
    main(args)
