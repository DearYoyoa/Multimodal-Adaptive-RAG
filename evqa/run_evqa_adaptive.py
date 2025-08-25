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
from torchvision import transforms
from classifier import Classifier1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def load_classifier(model_path):
    classifier = Classifier1().to(DEVICE)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def query_with_image(
        model,
        processor,
        classifier,
        image_path,
        screenshot_path,
        query_text,
        exp_dir='experiment/evqa/',
        sys_msg_filename_with_screenshot=None,
        sys_msg_filename_without_screenshot=None,
        data_source='url',
        use_adaptive=False
    ):
    with open(exp_dir + sys_msg_filename_with_screenshot, 'r') as f:
        query_system_msg_with_screenshot = f.read()
    with open(exp_dir + sys_msg_filename_without_screenshot, 'r') as f:
        query_system_msg_without_screenshot = f.read()

    query_image = load_image_wrapper(image_path, data_source='local')
    preprocessed_image = preprocess_image(query_image)

    # First query without screenshot
    query_system_msg = query_system_msg_without_screenshot
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
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[query_image], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    generated_output = model.generate(**inputs, max_new_tokens=1000, output_hidden_states=True, return_dict_in_generate=True)
    hidden_state = generated_output.hidden_states[-1][-1][:, 0, :].detach()

    # Use Classifier1 for prediction
    prediction = None
    use_screenshot = False
    if use_adaptive:
        with torch.no_grad():
            classifier_output = classifier(preprocessed_image, hidden_state)
            prediction = torch.argmax(classifier_output, dim=1).item()
        use_screenshot = (prediction == 1)
    else:
        use_screenshot = True  # Non-adaptive mode always uses screenshot

    if use_screenshot:
        query_system_msg = query_system_msg_with_screenshot
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
        generated_output = model.generate(**inputs, max_new_tokens=1000, return_dict_in_generate=True)
    else:
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

    generated_ids = generated_output.sequences
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts, messages_record, prediction, use_screenshot

def main(args):
    processor = AutoProcessor.from_pretrained("experiment/infoseek/models--HuggingFaceM4--idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "experiment/infoseek/models--HuggingFaceM4--idefics2-8b",
    ).to(DEVICE)
    classifier = load_classifier(args.classifier_path) if args.use_adaptive else None

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('data_1017/evqa_input.json', 'r') as f:
        data = json.load(f)
    
    samples = [sample for sample in data]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"data_1017/image/{sample['image_id']}.jpg"
        screenshot_path = f"data_1017/screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response, messages_record, prediction, use_screenshot = query_with_image(
            model,
            processor,
            classifier,
            image_path,
            screenshot_path,
            query_text,
            exp_dir='experiment/evqa/',
            sys_msg_filename_with_screenshot=args.sys_msg_filename_with_screenshot,
            sys_msg_filename_without_screenshot=args.sys_msg_filename_without_screenshot,
            data_source=args.data_source,
            use_adaptive=args.use_adaptive
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
                'classifier_prediction': prediction if args.use_adaptive else None,
                'use_screenshot': use_screenshot,
                'use_adaptive': args.use_adaptive
            }
        )

        # Use epoch value to distinguish filenames
        epoch = args.classifier_path.split('_')[-1].split('.')[0]  # Extract epoch value
        output_name = f'{args.output_root}/{args.log_name}_epoch_{epoch}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}_epoch_{epoch}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_with_screenshot', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_without_screenshot', type=str, required=True)
    # additional
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--data_source', type=str, default='local', choices=['url', 'local'])
    argparser.add_argument('--classifier_path', type=str, help='Path to the trained Classifier1 model')
    argparser.add_argument('--use_adaptive', action='store_true', help='Whether to use adaptive retrieval')

    args = argparser.parse_args()
    if args.use_adaptive and args.classifier_path is None:
        argparser.error("--use_adaptive requires --classifier_path")
    main(args) 