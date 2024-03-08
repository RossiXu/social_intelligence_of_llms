import numpy as np

from utils.utils import *
from dataset import *
from answer import get_exist_answers, answer_by_single_model


data_str2class = {'nq': NQ,
                  'mmlu': MMLU,
                  'bbh': BBH,
                  'winogrande': WinoGrande,
                  'race': Race,
                  'drop': Drop,
                  'gsm8k': GSM8K,
                  'math': Math,
                  'truthfulqa': TruthfulQA,
                  'sesi': SESI}


def eval(model,
         dataset,
         sample_num=-1,
         thread_num=4,
         chunk_size=15,
         seed=42,
         only_eval=False):
    # Path
    data = data_str2class[dataset](sample_num=sample_num)
    create_folder(data.data_path + '/input')
    data.input_path = data.data_path + f'/input/input_{sample_num}.csv' if sample_num > 0 else data.data_path + '/input.csv'
    create_folder(data.result_path)
    result_path = data.result_path + f'/{model}_{sample_num}_result.csv' if sample_num > 0 else data.result_path + f'/{model}_result.csv'
    paras = {"frequency_penalty": 0, "max_tokens": 200, "presence_penalty": 0, "temperature": 0.0, "top_p": 1, "seed": seed}

    # Reformat data
    if not os.path.exists(data.input_path):
        print(f'Create input file {data.input_path}.')
        data.reformat_data()

    # Answer
    if not only_eval:
        answer_by_single_model(model, data.input_path, result_path, chunk_size=chunk_size, thread_num=thread_num, paras=paras)

    # Score
    result = get_exist_answers(result_path)
    result['EA'] = result['PA'].apply(lambda x: data.extract_answer(x)).tolist()
    result['Label'] = result.apply(lambda x: data.score(x['CA'], x['EA'], x['PA']), axis=1).tolist()
    result.to_csv(result_path, index=False)

    stats = {data.score_name: np.mean(result['Label'])}
    print(stats)
    return stats
