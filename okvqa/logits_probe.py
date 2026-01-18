import json
import os
from collections import defaultdict
import argparse
import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import torch
from baukit import TraceDict

from probing_utils import extract_internal_reps_all_layers_and_tokens, load_model_and_validate_gpu, \
    probe_specific_layer_token, N_LAYERS, compile_probing_indices, LIST_OF_DATASETS, LIST_OF_MODELS, \
    MODEL_FRIENDLY_NAMES, get_indices_of_exact_answer, LIST_OF_PROBING_LOCATIONS, get_token_index, compute_metrics_probing, get_mlp_output
DEVICE = "cuda:0"
layers_to_trace = [f"model.text_model.layers.{i}.mlp" for i in range(32)]
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

    
def extract_exact_answer_features(model_output, question, inputs, model, tokenizer, pred, model_name, ground_truth, generated_output):
    # all_hidden_states = generated_output.hidden_states[-1]
    # model_hidden_state = model_output.hidden_states  # [batch_size, seq_len, hidden_dim]
    # print("model_hidden_state[0]:", model_hidden_state[0].shape)
    # print("model_hidden_state[-1]:", model_hidden_state[-1].shape)

    exact_tokens = {
        'exact_answer_last_token': [],
        # 'exact_answer_first_token': [], 
        # 'exact_answer_before_first_token': [],
        # 'exact_answer_after_last_token': []
    }

    generated_ids = generated_output.sequences[0]
    # full_tokens = generated_ids
    # full_answer_tokenized = tokenizer(pred, return_tensors="pt")['input_ids'][0]
    full_answer_tokenized = generated_output['sequences'][0]

    with torch.no_grad():
        with TraceDict(model, layers_to_trace, retain_input=True, clone=True) as ret:
            output = model(full_answer_tokenized.unsqueeze(dim=0), output_hidden_states=True)
    output_per_layer = get_mlp_output(ret, layers_to_trace)
    features = {}
    valid = 0
    t = -1
    exact_answer = None
    for answer in ground_truth:
        if answer in pred:
            valid = 1
            exact_answer = answer
            break
    # print("valid: ", valid)
    for token_type in exact_tokens.keys():
        layer_features = []
        t = get_token_index(token_type, tokenizer, question, model_name, 
                            full_answer_tokenized, exact_answer, valid, generated_ids)
        print("t: ", t)
        
    return t, valid
  

def query_with_image(
        model,
        processor,
        image_path,
        screenshot_path, 
        query_text,
        ground_truth,
        use_screenshot=False,
        exp_dir='experiment/okvqa/',
        sys_msg_filename=None,
        data_source='local',
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
    if use_screenshot:
        with torch.no_grad():
            model_output = model(
                **inputs,
                output_hidden_states=True, 
                return_dict=True, 
            )
            generated_output_with_screenshot = model.generate(
                **inputs, 
                max_new_tokens=1000, 
                output_hidden_states=True,
                return_dict_in_generate=True
            )
        generated_ids = generated_output_with_screenshot.sequences
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        hidden_state_with_screenshot = generated_output_with_screenshot.hidden_states[-1][-1][:, 0, :].detach().cpu().numpy()
        if not os.path.exists(f'hidden_okvqa/last_hidden_state_rir'):
            os.makedirs(f'hidden_okvqa/last_hidden_state_rir')
        np.save(f'hidden_okvqa/last_hidden_state_rir/{image_path.split("/")[-1].split(".")[0]}.npy', hidden_state_with_screenshot)
    else:
        with torch.no_grad():
            model_output = model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            generated_output = model.generate(
                **inputs,
                max_new_tokens=1000,
                output_scores=True,
                output_hidden_states=True, 
                return_dict_in_generate=True
            )
        
        # print("generated_output",generated_output['sequences'])
        # print("inputs:", inputs)
        # print("len(inputs): ", inputs[0])
        generated_ids = generated_output.sequences
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        pred = generated_texts[0].split('Assistant: ')[-1]

        pos, valid = extract_exact_answer_features(model_output,query_text, inputs, model, processor, pred, model.name_or_path, ground_truth, generated_output)
        
        logits_for_ex = torch.concatenate(generated_output['scores'])
        probas_for_ex = logits_for_ex.softmax(dim=1)
        # output_ids = generated_output['sequences'][0][len(inputs[0]):].cpu()
        output_ids = generated_output.sequences[0][len(inputs["input_ids"][0]):]
        logits_for_ex_generated_tokens = logits_for_ex.gather(1, output_ids.unsqueeze(1)).view(
            logits_for_ex.shape[0])
        probas_for_ex_generated_tokens = probas_for_ex.gather(1, output_ids.unsqueeze(1)).view(
            probas_for_ex.shape[0])

        if True:
            print("input_ids.shape[0]:", len(inputs["input_ids"][0]))
            print("pos: ", pos)
            print("logits_for_ex_generated_tokens:", logits_for_ex_generated_tokens)
            if valid==1:
                if torch.tensor(pos) - len(inputs["input_ids"][0]) >= 0:
                    exact_indices_for_logits = torch.tensor(pos) - len(inputs["input_ids"][0])
                else:
                    exact_indices_for_logits = -1
            else:
                exact_indices_for_logits = pos
            analyzed_logits = logits_for_ex_generated_tokens[exact_indices_for_logits]
            analyzed_probas = probas_for_ex_generated_tokens[exact_indices_for_logits]
            # logit_after_last_question_token = logits_for_ex_generated_tokens[
            #     min(exact_indices_for_logits[-1] + 1, len(logits_for_ex_generated_tokens) - 1)]
            # probas_after_last_question_token = probas_for_ex_generated_tokens[
            #     min(exact_indices_for_logits[-1] + 1, len(probas_for_ex_generated_tokens) - 1)]

            # all_logit_scores_after_last_token.append(logit_after_last_question_token.item())
            # all_probas_scores_after_last_token.append(probas_after_last_question_token.item())
        else:
            analyzed_logits = logits_for_ex_generated_tokens
            analyzed_probas = probas_for_ex_generated_tokens


        # generated_ids = generated_output.sequences
        # generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
        # pred = generated_texts[0].split('Assistant: ')[-1]


        return generated_texts, messages_record, analyzed_logits, analyzed_probas, 


def main(args):

    processor = AutoProcessor.from_pretrained("experiment/infoseek/models--HuggingFaceM4--idefics2-8b")
    model = AutoModelForVision2Seq.from_pretrained(
        "experiment/infoseek/models--HuggingFaceM4--idefics2-8b",
    ).to(DEVICE)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    # with open('data_1017/evqa_input.json', 'r') as f:
    #     data = json.load(f)
    with open('data_okvqa/okvqa_input.json', 'r') as f:
        data = json.load(f)

    # samples = [_ for cat_data in data.values() for _ in cat_data]
    samples = data
    # with open(f'{args.output_root}/samples.json', 'w') as f:
    #     json.dump(samples, f, indent=4)

    # run samples
    logs = []
    all_logit_scores_min = []
    all_logit_scores_max = []
    all_logit_scores_mean = []
    all_logit_scores_last_token = []
    all_probas_scores_min = []
    all_probas_scores_max = []
    all_probas_scores_mean = []
    all_probas_scores_last_token = []
    correctness = []


    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"data_okvqa/okvqa_image/{sample['image_id']}.jpg"
        screenshot_path = f"data_okvqa/screenshot/{sample['image_id']}-search_result.png"
        # image_path = f"local_data/infoseek_image/{sample['image_id']}"
        # screenshot_path = f"local_data/infoseek_screenshot/{sample['image_id']}-search_result.png"

        query_text = sample['question']
        ground_truth = sample['answer_eval']

        response, messages_record, analyzed_logits, analyzed_probas = query_with_image(
            model,
            processor,
            image_path,
            screenshot_path,
            query_text,
            ground_truth,
            use_screenshot=args.use_screenshot,
            exp_dir='experiment/okvqa/',
            sys_msg_filename=args.sys_msg_filename,
            data_source=args.data_source,
        )

        all_logit_scores_min.append(analyzed_logits.min().item())
        all_logit_scores_max.append(analyzed_logits.max().item())
        all_logit_scores_mean.append(analyzed_logits.mean().item())
        # all_logit_scores_last_token.append(analyzed_logits[-1].item())
        all_logit_scores_last_token.append(analyzed_logits.item())

        all_probas_scores_min.append(analyzed_probas.min().item())
        all_probas_scores_max.append(analyzed_probas.max().item())
        all_probas_scores_mean.append(analyzed_probas.mean().item())
        # all_probas_scores_last_token.append(analyzed_probas[-1].item())
        all_probas_scores_last_token.append(analyzed_probas.item())



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

        correctness.append(answer_in_pred)
    
    results = defaultdict(list)
    all_logit_scores_min = np.array(all_logit_scores_min)
    all_logit_scores_max = np.array(all_logit_scores_max)
    all_logit_scores_mean = np.array(all_logit_scores_mean)
    all_logit_scores_last_token = np.array(all_logit_scores_last_token)
    all_probas_scores_min = np.array(all_probas_scores_min)
    all_probas_scores_max = np.array(all_probas_scores_max)
    all_probas_scores_mean = np.array(all_probas_scores_mean)
    all_probas_scores_last_token = np.array(all_probas_scores_last_token)

    results["logits_min"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_logit_scores_min)
    results["logits_max"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_logit_scores_max)
    results["logits_mean"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_logit_scores_mean)
    results["logits_last_token"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_logit_scores_last_token)

    results["probas_min"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_probas_scores_min)
    results["probas_max"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_probas_scores_max)
    results["probas_mean"] = compute_metrics_probing(None, None, correctness, pos_label=1, predicted_probas=all_probas_scores_mean)
    results["probas_last_token"] = compute_metrics_probing(None, None, correctness, pos_label=1,  predicted_probas=all_probas_scores_last_token)

    output_name = f'{args.output_root}/logits_probas_{args.log_name}_exact_answer_last_token.json'
    with open(output_name, 'w') as f:
        json.dump(results, f, indent=4)
        # output_name = f'{args.output_root}/{args.log_name}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}.json'
        # with open(output_name, 'w') as f:
        #     json.dump(logs, f, indent=4)
    print("save models.")


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
