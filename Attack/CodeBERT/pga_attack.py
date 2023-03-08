'''For attacking CodeBERT models'''
import sys
import os
import json
import logging
import argparse
import warnings
import torch
import time
from run import set_seed
from run import TextDataset
import csv
from model import Model
from attacker import Attacker
import pdb

from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)



retval = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning) # Only report warning\

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

logger = logging.getLogger(__name__)

root_path = "your path"
adv_root_path = "your path"
task = "task"
model_name = "model"

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_type", default="train", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--do_robust", action='store_true')
    parser.add_argument("--attack", default="our", type=str)
    parser.add_argument("--index", nargs='+', default=[0, 500],
                        help="Optional input sequence length after tokenization.")

    ## Other parameters
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="microsoft/codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--number_labels", type=int, default=250,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")

    args = parser.parse_args()

    args.device = torch.device("cuda")
    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    if args.data_type == "train":
        tgt_model_file = f"{root_path}/models/{model_name}/{task}/model/model.bin"
        args.csv_root_path = f"{adv_root_path}/results/{model_name}/{task}"
        args.eval_data_file = f"{adv_root_path}/datasets/{model_name}/{task}/{args.data_type}_subs_{int(args.index[0])}_{int(args.index[1])}_our.jsonl"
    else:
        args.eval_data_file = f"{root_path}/datasets/{model_name}/{task}/{args.data_type}_subs_{int(args.index[0])}_{int(args.index[1])}_our.jsonl"
        if args.do_robust:
            tgt_model_file = f"{adv_root_path}/models/{model_name}/{task}/model/adv_model_{args.attack}.bin"
            args.csv_root_path = f"{adv_root_path}/results/{model_name}/{task}"
        else:
            tgt_model_file = f"{root_path}/models/{model_name}/{task}/model/model.bin"
            args.csv_root_path = f"{root_path}/results/{model_name}/{task}"

    if not os.path.isdir(args.csv_root_path):
        os.makedirs(args.csv_root_path)

    if args.do_robust:
        csv_file_name = f"{args.csv_root_path}/pga_test_{int(args.index[0])}_{int(args.index[1])}_{args.attack}.csv"
    else:
        csv_file_name = f"{args.csv_root_path}/pga_{int(args.index[0])}_{int(args.index[1])}.csv"

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = args.number_labels  # 只有一个label?
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    model = Model(model, config, tokenizer, args)
    model.load_state_dict(torch.load(tgt_model_file))
    model.to(args.device)

    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
    tokenizer_mlm = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
    codebert_mlm.to('cuda')

    ## Load Dataset
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    source_codes = []
    substs = []
    with open(args.eval_data_file) as rf:
        for line in rf:
            item = json.loads(line.strip())
            source_codes.append(item["func"].replace("\\n", "\n").replace('\"', '"'))
            substs.append(item["substitutes"])
    assert (len(source_codes) == len(eval_dataset) == len(substs))

    success_attack = 0
    total_cnt = 0

    attacker = Attacker(args, model, tokenizer, codebert_mlm, tokenizer_mlm, use_bpe=1, threshold_pred_score=0)
    query_times = 0
    attack_type = "PGA"
    results = []
    for index, example in enumerate(eval_dataset):
        example_start_time = time.time()
        code = source_codes[index]
        subs = substs[index]
        try:
            code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words = attacker.ga_attack(example, code, subs)
        except:
            continue
        example_end_time = (time.time() - example_start_time) / 60
        score_info = ''
        if names_to_importance_score is not None:
            for key in names_to_importance_score.keys():
                score_info += key + ':' + str(names_to_importance_score[key]) + ','

        replace_info = ''
        if replaced_words is not None:
            for key in replaced_words.keys():
                replace_info += key + ':' + replaced_words[key] + ','
        # print("Query times in this attack: ", model.query - query_times)
        # print("All Query times: ", model.query)
        results.append([index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type, model.query - query_times, example_end_time])
        query_times = model.query

        if is_success >= -1:
            # 如果原来正确
            total_cnt += 1
        if is_success == 1:
            success_attack += 1
            print(f"{index} successful")

        if total_cnt == 0:
            continue
    writer = csv.writer(open(csv_file_name, 'w'))
    writer.writerow(["Index", "Original Code", "Program Length", "Adversarial Code", "True Label", "Original Prediction", "Adv Prediction", "Is Success", "Extracted Names", "Importance Score", "No. Changed Names", "No. Changed Tokens", "Replaced Names", "Attack Type", "Query Times", "Time Cost"])
    for result in results:
        writer.writerow(result)


if __name__ == '__main__':
    main()
