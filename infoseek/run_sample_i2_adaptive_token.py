import json
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from classifier_token_probe import MLPClassifier
import logging

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_classifier(model_path, input_dim, num_classes):
    classifier = MLPClassifier(input_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier

def extract_vision_features(model_output, inputs, model, layer_idx):

    layer_hidden_states = model_output.hidden_states[layer_idx]

    image_token_id = model.config.image_token_id
    input_ids = inputs['input_ids'][0]

    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]

    gaps = image_token_positions[1:] - image_token_positions[:-1]
    image_boundaries = [0] + ((gaps > 2).nonzero(as_tuple=True)[0] + 1).tolist() + [len(image_token_positions)]

    vision_hidden_states = []
    for i in range(len(image_boundaries) - 1):
        start_idx = image_token_positions[image_boundaries[i]]
        end_idx = image_token_positions[image_boundaries[i+1] - 1]

        image_patches = layer_hidden_states[:, start_idx:end_idx+1, :]
        image_feature = image_patches.mean(dim=1)
        vision_hidden_states.append(image_feature)

    vision_features = torch.stack(vision_hidden_states)  # [num_images, batch_size, hidden_dim]
    return vision_features

def query_with_image(
        model,
        processor,
        classifier,
        image_path,
        screenshot_path,
        query_text,
        exp_dir='experiment/infoseek/',
        sys_msg_filename_with_screenshot=None,
        sys_msg_filename_without_screenshot=None,
        use_adaptive=False,
        layer_idx=0,
        classifier_path=None
    ):
    with open(exp_dir + sys_msg_filename_with_screenshot, 'r') as f:
        query_system_msg_with_screenshot = f.read()
    with open(exp_dir + sys_msg_filename_without_screenshot, 'r') as f:
        query_system_msg_without_screenshot = f.read()

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
    inputs = processor(text=prompt, images=[Image.open(image_path)], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():

        model_output = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    image_hidden_state = extract_vision_features(model_output, inputs, model, layer_idx)
    original_vision_features = model_output.image_hidden_states.reshape(2, -1, 4096).mean(dim=1)

    del model_output
    torch.cuda.empty_cache()
    generated_output = model.generate(
        **inputs, 
        max_new_tokens=1000, 
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    hidden_state = generated_output.hidden_states[-1][-1][:, 0, :].detach()

    prediction = None
    use_screenshot = False
    if use_adaptive:
        with torch.no_grad():

            if 'original' in classifier_path:
                x = original_vision_features[0].unsqueeze(0).to(DEVICE)
            else:
                x = image_hidden_state[0].unsqueeze(0).to(DEVICE)
            
            x = torch.cat((x.reshape(-1, x.shape[-1]), hidden_state), dim=1)
            classifier_output = classifier(x)
            prediction = torch.argmax(classifier_output, dim=1).item()
        use_screenshot = (prediction == 1)
    else:
        use_screenshot = True

    if use_screenshot:
        query_system_msg = query_system_msg_with_screenshot
        screenshot_image = Image.open(screenshot_path)
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
        inputs = processor(text=prompt, images=[Image.open(image_path), screenshot_image], return_tensors="pt")
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

    layer_idx = int(args.classifier_path.split('layer_')[1].split('_')[0])
    input_dim = 8192  
    classifier = load_classifier(args.classifier_path, input_dim, 2) if args.use_adaptive else None

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('local_data/infoseek_data.json', 'r') as f:
        data = json.load(f)
    
    samples = [_ for cat_data in data.values() for _ in cat_data]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # run samples
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"local_data/infoseek_image/{sample['image_id']}"
        screenshot_path = f"local_data/infoseek_screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response, messages_record, prediction, use_screenshot = query_with_image(
            model,
            processor,
            classifier,
            image_path,
            screenshot_path,
            query_text,
            exp_dir='experiment/infoseek/',
            sys_msg_filename_with_screenshot=args.sys_msg_filename_with_screenshot,
            sys_msg_filename_without_screenshot=args.sys_msg_filename_without_screenshot,
            use_adaptive=args.use_adaptive,
            layer_idx=layer_idx,
            classifier_path=args.classifier_path
        )

        pred = response[0].split('Assistant: ')[-1]
        if isinstance(sample['answer'], list):
            answer_in_pred = any(_.lower() in pred.lower() for _ in sample['answer'])
        else:
            answer_in_pred = sample['answer'].lower() in pred.lower()

        logs.append({
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
        })

        epoch = args.classifier_path.split('_')[-1].split('.')[0]
        output_name = f'{args.output_root}/{args.log_name}_epoch_{epoch}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}_epoch_{epoch}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)

def check_existing_logs(args):
    # Extract epoch from classifier path
    epoch = args.classifier_path.split('_')[-1].split('.')[0] if args.classifier_path else 'no_classifier'
    
    # Construct log filename
    log_filename = (f'{args.output_root}/{args.log_name}_epoch_{epoch}_{args.idx_offset}.json' 
                   if args.idx_offset != 0 
                   else f'{args.output_root}/{args.log_name}_epoch_{epoch}.json')
    
    if os.path.exists(log_filename):
        logging.warning(f"Log file already exists: {log_filename}")
        return True
    return False

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_with_screenshot', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_without_screenshot', type=str, required=True)
    # additional
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--classifier_path', type=str, help='Path to the trained token probe classifier model')
    argparser.add_argument('--use_adaptive', action='store_true', help='Whether to use adaptive retrieval')

    args = argparser.parse_args()
    if args.use_adaptive and args.classifier_path is None:
        argparser.error("--use_adaptive requires --classifier_path")

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Check for existing logs before running main
    if not check_existing_logs(args):
        logging.info(f"Running experiment {args.log_name} with idx_offset {args.idx_offset}")
        main(args)
    else:
        logging.info("Skipping execution as log file already exists") 
    # main(args)
