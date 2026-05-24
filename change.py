import json  
import sys
import os
sys.path.append('/home/derin/Documents/syntax_finetuning/src')

from conduct_evaluation import conduct_evaluation
from config import DatasetConfig
from inference_functions.tree_decoder import TreeDecoder

#with open('config_inference_languages_logits_grctrestr_qwen_base_original.yaml', 'r') as file:
#    configs = yaml.safe_load(file)
    
    
dataset_dict = [{'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Chinese-GSD/zh_gsd_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Chinese-GSD/zh_gsd_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Chinese-GSD/zh_gsd_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_Chinese-GSD/zh_gsd-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'zh_gsd_grct'}, {'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Japanese-GSD/ja_gsd_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Japanese-GSD/ja_gsd_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Japanese-GSD/ja_gsd_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_Japanese-GSD/ja_gsd-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'ja_gsd_grct'}, {'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_English-EWT/en_ewt_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_English-EWT/en_ewt_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_English-EWT/en_ewt_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_English-EWT/en_ewt-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'en_ewt_grct'}, {'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Finnish-FTB/fi_ftb_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Finnish-FTB/fi_ftb_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Finnish-FTB/fi_ftb_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_Finnish-FTB/fi_ftb-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'fi_ftb_grct'}, {'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_French-GSD/fr_gsd_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_French-GSD/fr_gsd_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_French-GSD/fr_gsd_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_French-GSD/fr_gsd-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'fr_gsd_grct'}, {'train_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Russian-SynTagRus/ru_simple_syntagrus_grct_train.json', 'dev_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Russian-SynTagRus/ru_simple_syntagrus_grct_dev.json', 'test_file_path': '/home/derin/Documents/syntax_finetuning_data/dataset_simple_rel/UD_Russian-SynTagRus/ru_simple_syntagrus_grct_test.json', 'conll_test_file_path': '/home/derin/Documents/syntax_finetuning_data/conllu/UD_Russian-SynTagRus/ru_syntagrus-ud-test.conllu', 'treebank_repr': 'grct', 'treebank': 'ru_syntagrus_grct'}]

    
datasets = {d['treebank']: DatasetConfig(d) for d in dataset_dict}

dec = TreeDecoder('grct')

for file_dir in ['main_logits', 'original_logits', 'root_logits']:
 for pred_filename in os.listdir(file_dir):
  #file_dir = 'main_logits'
  #pred_filename = 'Qwen3_8B_zh_gsd_grct_zh_gsd_grct_42_main_logits_2.jsonl'
  if ".jsonl" in pred_filename and "metrics" not in pred_filename:
    pred_trees = []
    cnt = 0
    with open(f"{file_dir}/{pred_filename}", 'r', encoding='utf-8') as f:
      #print(f.read(30))
      print(f"{file_dir}/{pred_filename}")
      for line_num, line in enumerate(f): 
        #print(line)
        item = json.loads(line)
        old_pred_tree = item['pred_tree']
        item['pred_tree'] = dec.decode_tree(item['pred_output'],check_seq=True)
        if old_pred_tree != item['pred_tree']:
            cnt += 1
            #print(old_pred_tree)
            #print(item['pred_tree'])
            #print(item['pred_output'])
            #print()
        #1 / 0
        pred_trees.append(item)
    print(pred_filename, cnt, flush=True)
    #1 / 0
    result_path = f'{file_dir}/{pred_filename}'
    with open(result_path, 'w', encoding='utf-8') as json_file:
      for s_i in range(len(pred_trees)):
        json_file.write(json.dumps(pred_trees[s_i],
            ensure_ascii=False) + '\n')
      json_file.flush()
    tr = '_'.join(pred_filename.split('B_')[1].split('_42')[0].split('_')[:3])
    dataset_config = datasets[tr]
    print(tr)
    conduct_evaluation(file_dir, dataset_config, result_path, "metric")
