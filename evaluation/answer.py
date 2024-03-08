import pandas as pd
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import openai

openai_chat_models = ['gpt-3.5-turbo',
                      'gpt-3.5-turbo-0613',
                      'gpt-3.5-turbo-16k',
                      'gpt-3.5-turbo-16k-0613',
                      'gpt3.5-turbo-0301',
                      'gpt-4-0613']
openai_completion_models = ['text-davinci-003',
                            'text-davinci-002',
                            'text-davinci-001',
                            'text-curie-001',
                            'text-babbage-001',
                            'text-ada-001',
                            'davinci',
                            'curie',
                            'babbage',
                            'ada']


def get_exist_answers(answer_path):
    try:
        if os.path.exists(answer_path):
            exsit_answers = pd.read_csv(answer_path)
            condition = exsit_answers['PA'].apply(lambda x: bool(type(x) is str and x.strip() and x.strip() != 'None'))
            exsit_answers = exsit_answers[condition]
            exsit_answers = exsit_answers[['ID', 'CA', 'PA']]
        else:
            exsit_answers = []
    except:
        exsit_answers = []
    return exsit_answers


def _chat_with_llm(model, message, paras):
    if model in openai_chat_models:
        message = [{"role": "user", "content": message}] if type(message) is str else message
        completion = openai.ChatCompletion.create(
            model=model,
            messages=message,
            **paras
        )
        answer = completion['choices'][0]['message']['content']
        return answer
    elif model in openai_completion_models:
        completion = openai.Completion.create(
            model=model,
            prompt=message,
            **paras
        )
        answer = completion['choices'][0]['text']
        return answer
    else:
        # Chat with other models
        # To be filled
        print(f'Please download model {model} and renew the inference code.')
        answer = None
        return answer


def chat_with_llm(model, message,
                  addition_information=None,
                  retry_time=3, run_time=20, paras={}, port=9222):
    """
    answer = [model_answer: str, additional_information: list]
    """
    # Retry, if it fails or times out
    answer = [] if type(message) is list else 'None'
    for retry_idx in range(retry_time):
        message = message[len(answer):] if type(message) is list else message
        try:
            if type(message) is list:
                for single_message in message:
                    answer.append(_chat_with_llm(model, single_message, paras))
            else:
                answer = _chat_with_llm(model, message, paras)
            break
        except Exception as e:
            print(e)
            print('Failed to chat with %s for the %s time.' % (model, str(retry_idx + 1)))

    if addition_information:
        if type(answer) is str:
            answer = [answer, addition_information]
        else:
            answer = [[a, addition_information[a_idx]] for a_idx, a in enumerate(answer)]
    return answer


def chat_with_llm_multi_thread(messages,
                               model='gpt-3.5-turbo',
                               addition_information=None,
                               chunk_size=5, thread_num=8,
                               retry_time=3, run_time=20,
                               paras={}):
    if not addition_information:
        addition_information = [None] * len(messages)
    with ThreadPoolExecutor(max_workers=thread_num) as executor:
        ports = [[9222 + i, True] for i in range(thread_num)]
        futures = [executor.submit(chat_with_llm, model, messages[start_idx:start_idx + chunk_size],
                                   addition_information[start_idx:start_idx + chunk_size], retry_time,
                                   run_time, paras, ports) for start_idx in range(0, len(messages), chunk_size)]
        for future in tqdm(concurrent.futures.as_completed(futures)):
            # try:
                answer = future.result()
                yield answer
            # except Exception as e:
            #     print(f"An error occurred: {e}")


def answer_by_single_model(model,
                           test_path, answer_path,
                           save=True,
                           output=True,
                           chunk_size=32, thread_num=8,
                           retry_time=3,
                           run_time=20,
                           paras={}):
    print('Start test model %s!' % model)

    # Read tests
    tests = pd.read_csv(test_path)
    try:
        exsit_answers = get_exist_answers(answer_path)

        answers = exsit_answers.to_numpy().tolist()
        condition = tests['ID'].apply(lambda x: x not in exsit_answers['ID'].tolist())
        rest_tests = tests[condition].reset_index(drop=True)

        print(f'Total instances: {len(tests)}. Left instances: {len(rest_tests)}.')
    except:
        answers = []
        rest_tests = tests

        print(f'Total instances: {len(tests)}.')

    # Answer
    post_ids = rest_tests['ID'].tolist()
    cas = rest_tests['CA'].tolist()
    inputs = rest_tests['Input'].tolist()
    addition_information = list(zip(post_ids, cas))
    for model_answers in tqdm(chat_with_llm_multi_thread(inputs, model, addition_information, chunk_size, thread_num, retry_time, run_time, paras)):
        for idx, model_answer in enumerate(model_answers):
            pa = model_answer[0]
            post_id, ca = model_answer[1]
            answers.append([post_id, ca, pa])
        if save:
            data = pd.DataFrame(answers, columns=['ID', 'CA', 'PA'])
            data.to_csv(answer_path, index=False)
            if output:
                print(f'Successfully answer {len(list(set([a[0] for a in answers])))} questions!')
    return answers