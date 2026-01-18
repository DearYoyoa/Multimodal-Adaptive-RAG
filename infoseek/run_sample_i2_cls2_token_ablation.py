import json
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from classifier_token_probe_ablation import MLPClassifier

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_classifier(model_path, input_dim, num_classes):
    classifier = MLPClassifier(input_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier


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
        use_rir=False,
        classifier_path=None
    ):
    with open(exp_dir + sys_msg_filename_with_screenshot, 'r') as f:
        query_system_msg_with_screenshot = f.read()
    with open(exp_dir + sys_msg_filename_without_screenshot, 'r') as f:
        query_system_msg_without_screenshot = f.read()

    # Query without screenshot
    query_system_msg = query_system_msg_without_screenshot
    messages_without_screenshot = [
        {
            "role": "user",
            "content": [
                {'type': 'text', 'text': query_system_msg},
                {"type": "image"},
                {"type": "text", "text": "Query: " + query_text},
            ]
        }
    ]
    prompt_without_screenshot = processor.apply_chat_template(messages_without_screenshot, add_generation_prompt=True)
    inputs_without_screenshot = processor(text=prompt_without_screenshot, images=[Image.open(image_path)], return_tensors="pt")
    inputs_without_screenshot = {k: v.to(DEVICE) for k, v in inputs_without_screenshot.items()}

    with torch.no_grad():
        # model_output_without_screenshot = model(
        #     **inputs_without_screenshot,
        #     output_hidden_states=True,
        #     return_dict=True,
        # )

        generated_output_without_screenshot = model.generate(
            **inputs_without_screenshot, 
            max_new_tokens=1000, 
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    hidden_state_without_screenshot = generated_output_without_screenshot.hidden_states[-1][-1][:, 0, :].detach()

    # Query with screenshot
    query_system_msg = query_system_msg_with_screenshot
    context_text = ("In the screenshot, the large image on the left is the query image for a reverse image search. "
                    "The smaller images on the right and their titles are the top hits from the search. ")
    messages_with_screenshot = [
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
    prompt_with_screenshot = processor.apply_chat_template(messages_with_screenshot, add_generation_prompt=True)
    inputs_with_screenshot = processor(text=prompt_with_screenshot, images=[Image.open(image_path), Image.open(screenshot_path)], return_tensors="pt")
    inputs_with_screenshot = {k: v.to(DEVICE) for k, v in inputs_with_screenshot.items()}

    with torch.no_grad():
        generated_output_with_screenshot = model.generate(
            **inputs_with_screenshot, 
            max_new_tokens=1000, 
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    hidden_state_with_screenshot = generated_output_with_screenshot.hidden_states[-1][-1][:, 0, :].detach()

    # Use classifier for prediction
    with torch.no_grad():
        if use_rir:
            text_hidden = torch.cat([
                hidden_state_without_screenshot,
                hidden_state_with_screenshot
            ], dim=1)
            x = text_hidden
        else:
            x = hidden_state_without_screenshot
        
        classifier_output = classifier(x)
        prediction = torch.argmax(classifier_output, dim=1).item()

    generated_ids_without_screenshot = generated_output_without_screenshot.sequences
    generated_texts_without_screenshot = processor.batch_decode(generated_ids_without_screenshot, skip_special_tokens=True)

    generated_ids_with_screenshot = generated_output_with_screenshot.sequences
    generated_texts_with_screenshot = processor.batch_decode(generated_ids_with_screenshot, skip_special_tokens=True)

    return generated_texts_without_screenshot, generated_texts_with_screenshot, messages_without_screenshot, messages_with_screenshot, prediction

def main(args):
    processor = AutoProcessor.from_pretrained("experiment/infoseek/models--HuggingFaceM4--idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "experiment/infoseek/models--HuggingFaceM4--idefics2-8b",
    ).to(DEVICE)

    use_rir = 'rir' in args.classifier_path and 'wo_rir' not in args.classifier_path
    input_dim = 8192 if use_rir else 4096
    classifier = load_classifier(args.classifier_path, input_dim, 4)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('data_okvqa/okvqa_input.json', 'r') as f:
        data = json.load(f)
    
    # samples = [_ for cat_data in data.values() for _ in cat_data]
    samples = [sample for sample in data]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)

    # run samples
    logs_1 = []
    logs_2 = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"data_okvqa/okvqa_image/{sample['image_id']}.jpg"
        screenshot_path = f"data_okvqa/screenshot/{sample['image_id']}-search_result.png"
        query_text = sample['question']
        response_without_screenshot, response_with_screenshot, messages_without_screenshot, messages_with_screenshot, prediction = query_with_image(
            model,
            processor,
            classifier,
            image_path,
            screenshot_path,
            query_text,
            exp_dir='experiment/infoseek/',
            sys_msg_filename_with_screenshot=args.sys_msg_filename_with_screenshot,
            sys_msg_filename_without_screenshot=args.sys_msg_filename_without_screenshot,
            use_rir=use_rir,
            classifier_path=args.classifier_path
        )

        pred_without_screenshot = response_without_screenshot[0].split('Assistant: ')[-1]
        pred_with_screenshot = response_with_screenshot[0].split('Assistant: ')[-1]

        if isinstance(sample['answer'], list):
            answer_in_pred_without_screenshot = any(_.lower() in pred_without_screenshot.lower() for _ in sample['answer'])
            answer_in_pred_with_screenshot = any(_.lower() in pred_with_screenshot.lower() for _ in sample['answer'])
        else:
            answer_in_pred_without_screenshot = sample['answer'].lower() in pred_without_screenshot.lower()
            answer_in_pred_with_screenshot = sample['answer'].lower() in pred_with_screenshot.lower()

        use_screenshot_1 = prediction != 1
        pred_1 = pred_with_screenshot if use_screenshot_1 else pred_without_screenshot
        answer_in_pred_1 = answer_in_pred_with_screenshot if use_screenshot_1 else answer_in_pred_without_screenshot
        messages_1 = messages_with_screenshot if use_screenshot_1 else messages_without_screenshot
        response_1 = response_with_screenshot if use_screenshot_1 else response_without_screenshot

        use_screenshot_2 = prediction == 2
        pred_2 = pred_with_screenshot if use_screenshot_2 else pred_without_screenshot
        answer_in_pred_2 = answer_in_pred_with_screenshot if use_screenshot_2 else answer_in_pred_without_screenshot
        messages_2 = messages_with_screenshot if use_screenshot_2 else messages_without_screenshot
        response_2 = response_with_screenshot if use_screenshot_2 else response_without_screenshot

        log_entry_1 = {
            'idx': idx,
            'answer_in_pred': answer_in_pred_1,
            'data_id': sample['data_id'],
            'image_id': sample['image_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'pred': pred_1,
            'answer_eval': sample['answer_eval'],
            'full_messages_record': str(messages_1),
            'full_response': str(response_1),
            'classifier_prediction': prediction,
            'use_screenshot': use_screenshot_1,
            'use_adaptive': True
        }

        log_entry_2 = {
            'idx': idx,
            'answer_in_pred': answer_in_pred_2,
            'data_id': sample['data_id'],
            'image_id': sample['image_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'pred': pred_2,
            'answer_eval': sample['answer_eval'],
            'full_messages_record': str(messages_2),
            'full_response': str(response_2),
            'classifier_prediction': prediction,
            'use_screenshot': use_screenshot_2,
            'use_adaptive': True
        }

        logs_1.append(log_entry_1)
        logs_2.append(log_entry_2)

        epoch = args.classifier_path.split('_')[-1].split('.')[0]
        output_name_1 = f'{args.output_root}/{args.log_name}_1_epoch_{epoch}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}_1_epoch_{epoch}.json'
        output_name_2 = f'{args.output_root}/{args.log_name}_2_epoch_{epoch}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}_2_epoch_{epoch}.json'
        
        with open(output_name_1, 'w') as f:
            json.dump(logs_1, f, indent=4)
        
        with open(output_name_2, 'w') as f:
            json.dump(logs_2, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_with_screenshot', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_without_screenshot', type=str, required=True)
    # additional
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--classifier_path', type=str, required=True, help='Path to the trained token probe classifier model')

    args = argparser.parse_args()
    main(args)
