import json
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from classifier_token_probe_okvqa import MLPClassifier


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_classifier(model_path, input_dim, num_classes):
    classifier = MLPClassifier(input_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier

def extract_vision_features(model_output, inputs, model, layer_idx):
    """
    从模型输出中提取指定层的视觉特征。
    """
    # 获取指定层的hidden states
    layer_hidden_states = model_output.hidden_states[layer_idx]
    
    # 找到图像token的位置
    image_token_id = model.config.image_token_id
    input_ids = inputs['input_ids'][0]
    
    # 找到所有图像token的位置
    image_token_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    
    # 找到gap大于2的位置，这些位置表示不同图片的边界
    gaps = image_token_positions[1:] - image_token_positions[:-1]
    image_boundaries = [0] + ((gaps > 2).nonzero(as_tuple=True)[0] + 1).tolist() + [len(image_token_positions)]
    
    # 获取每个图像的特征
    vision_hidden_states = []
    for i in range(len(image_boundaries) - 1):
        start_idx = image_token_positions[image_boundaries[i]]
        end_idx = image_token_positions[image_boundaries[i+1] - 1]
        
        # 获取这张图片所有patch的hidden states
        image_patches = layer_hidden_states[:, start_idx:end_idx+1, :]
        # 对所有patch取平均得到这张图片的表征
        image_feature = image_patches.mean(dim=1)
        vision_hidden_states.append(image_feature)
    
    # 堆叠所有图像的特征
    vision_features = torch.stack(vision_hidden_states)  # [num_images, batch_size, hidden_dim]
    return vision_features

def query_with_image(
        # model,
        # processor,
        classifier,
        image_path,
        screenshot_path,
        query_text,
        exp_dir='experiment/evqa/',
        # sys_msg_filename_with_screenshot=None,
        sys_msg_filename_without_screenshot=None,
        # layer_idx=0,
        classifier_path=None,
        last_hidden_without_screenshot=None,
        image_hidden_state=None,
        original_vision_features=None
    ):
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
    hidden_state_without_screenshot = torch.tensor(last_hidden_without_screenshot).to(DEVICE)
    image_hidden_state = torch.tensor(image_hidden_state).to(DEVICE)
    # image_hidden_state = image_hidden_state[layer_idx]
    original_vision_features = torch.tensor(original_vision_features).to(DEVICE)
    # Use classifier for prediction
    with torch.no_grad():
        # 只使用第一个图像的表征
        if 'original' in classifier_path:
            x = original_vision_features[0].unsqueeze(0).to(DEVICE)
        else:
            x = image_hidden_state[0].unsqueeze(0).to(DEVICE)
        x = torch.cat((x, hidden_state_without_screenshot), dim=1)
        # x = hidden_state_without_screenshot
        # print("x.shape:", x.shape)
        # print("hidden_state_without_screenshot.shape:", hidden_state_without_screenshot.shape)
        
        classifier_output = classifier(x)
        prediction = torch.argmax(classifier_output, dim=1).item()
    return messages_without_screenshot, prediction

    # return generated_texts_without_screenshot, generated_texts_with_screenshot, messages_without_screenshot, messages_with_screenshot, prediction

def main(args):
    # processor = AutoProcessor.from_pretrained("experiment/infoseek/models--HuggingFaceM4--idefics2-8b")
    # model = AutoModelForVision2Seq.from_pretrained(
    #     "experiment/infoseek/models--HuggingFaceM4--idefics2-8b",
    # ).to(DEVICE)

    # 从模型路径中提取layer_idx和是否使用rir
    print("classifier_path: ", args.classifier_path)
    # layer_idx = int(args.classifier_path.split('layer_')[1].split('_')[0])
    input_dim = 4096
    classifier = load_classifier(args.classifier_path, input_dim, 2)

    # initial setup
    os.makedirs(f'{args.output_root}/', exist_ok=True)

    # load sample data
    with open('data_1017/evqa_input.json', 'r') as f:
        data = json.load(f)

    # samples = [_ for cat_data in data.values() for _ in cat_data]
    samples = [sample for sample in data]
    with open(f'{args.output_root}/samples.json', 'w') as f:
        json.dump(samples, f, indent=4)
    
    # with open('data_1017/evqa/evqa_i2_test/logs_evqa_i2_test.json', 'r') as f:
    with open('data_1017/evqa/evqa_i2_test/logs_evqa_i2_test.json', 'r') as f:
        test_data = json.load(f)
    
    # run samples
    total_samples = len(data)
    correct = 0
    logs = []
    for idx, sample in tqdm(enumerate(samples[args.idx_offset:]), total=len(samples[args.idx_offset:])):
        idx = args.idx_offset + idx
        image_path = f"data_1017/image/{sample['image_id']}.jpg"
        screenshot_path = f"data_1017/screenshot/{sample['image_id']}-search_result.png"
        last_hidden = np.load(f"hidden_state_i2_test/hidden_evqa/last_hidden_state/{sample['image_id']}.npy")
        image_hidden = np.load(f"hidden_state_i2_test/hidden_evqa/hidden_state/{sample['image_id']}.npy")
        original_hidden = np.load(f"hidden_state_i2_test/hidden_evqa/original_hidden_state/{sample['image_id']}.npy")
        # image_path = f"local_data/infoseek_image/{sample['image_id']}"
        # screenshot_path = f"local_data/infoseek_screenshot/{sample['image_id']}-search_result.png"
        # last_hidden = np.load(f"hidden_state_i2_test/hidden_infoseek/last_hidden_state/{sample['image_id'].split('.')[0]}.npy")
        # image_hidden = np.load(f"hidden_state_i2_test/hidden_infoseek/hidden_state/{sample['image_id'].split('.')[0]}.npy")
        # original_hidden = np.load(f"hidden_state_i2_test/hidden_infoseek/original_hidden_state/{sample['image_id'].split('.')[0]}.npy")
        query_text = sample['question']
        messages_without_screenshot, prediction = query_with_image(
            # model,
            # processor,
            classifier,
            image_path,
            screenshot_path,
            query_text,
            exp_dir='experiment/evqa/',
            # sys_msg_filename_with_screenshot=args.sys_msg_filename_with_screenshot,
            sys_msg_filename_without_screenshot=args.sys_msg_filename_without_screenshot,
            # layer_idx=layer_idx,
            classifier_path=args.classifier_path,
            last_hidden_without_screenshot=last_hidden,
            image_hidden_state=image_hidden,
            original_vision_features=original_hidden
        )
        answer_in_pred = None
        for d in test_data:
            if d["image_id"] == sample["image_id"]:
                answer_in_pred = d["answer_in_pred"]
                break

        if prediction == 1:
            if answer_in_pred is True:
                correct += 1
        elif prediction == 0:
            if answer_in_pred is False:
                correct += 1

        log_entry = {
            'idx': idx,
            'data_id': sample['data_id'],
            'image_id': sample['image_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'answer_eval': sample['answer_eval'],
            'answer_in_pred': answer_in_pred,
            'classifier_prediction': prediction,
        }
        logs.append(log_entry)
        epoch = args.classifier_path.split('_')[-1].split('.')[0]
        output_name = f'{args.output_root}/{args.log_name}_epoch_{epoch}_{args.idx_offset}.json' if args.idx_offset != 0 else f'{args.output_root}/{args.log_name}_epoch_{epoch}.json'
        with open(output_name, 'w') as f:
            json.dump(logs, f, indent=4)
    correctness = correct / total_samples
    with open(f'{args.output_root}/{args.log_name}_accuracy.json', 'w') as f:
        json.dump({
            'probe_accuracy': correctness
        }, f, indent=4)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # basics
    argparser.add_argument('--output_root', type=str, required=True)
    argparser.add_argument('--log_name', type=str, required=True)
    # argparser.add_argument('--sys_msg_filename_with_screenshot', type=str, required=True)
    argparser.add_argument('--sys_msg_filename_without_screenshot', type=str, required=True)
    # additional
    argparser.add_argument('--idx_offset', type=int, required=True)
    argparser.add_argument('--classifier_path', type=str, required=True, help='Path to the trained token probe classifier model')

    args = argparser.parse_args()
    main(args)

