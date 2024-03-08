from abc import ABC, abstractmethod
from collections import Counter
import pandas as pd
import random
import re

from utils.match import exact_match, fuzzy_match, normalize
from utils.utils import *
from utils.math_util import *


class Dataset(ABC):
    def __init__(self, data_path, result_path, sample_num):
        self.data_path = data_path
        self.result_path = result_path
        self.sample_num = sample_num

    @abstractmethod
    def reformat_data(self):
        pass

    @abstractmethod
    def extract_answer(self, text):
        pass

    @abstractmethod
    def score(self, ca, ea, pa):
        pass


class NQ(Dataset):
    def __init__(self, data_path='benchmark/natural_question', result_path='result/natural_question', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'Please answer the question: {question}'
        self.score_name = 'em'

    def reformat_data(self):
        data = pd.read_csv(self.data_path + '/data/validation.csv')
        data['ID'] = range(len(data))
        data['Input'] = data.apply(lambda x: self.prompt.format(question=x['Question']), axis=1).tolist()
        data.drop('Question', axis=1, inplace=True)
        data = data[['ID', 'Input', 'Answer']]
        data.rename(columns={'Answer': 'CA'}, inplace=True)
        if self.sample_num >= 1:
            print(f'Sample {self.sample_num} instances from {len(data)} instances!')
            data = data.sample(self.sample_num)
        elif 0 < self.sample_num < 1:
            print(f'Sample {int(self.sample_num * len(data))} instances from {len(data)} instances!')
            data = data.sample(int(self.sample_num * len(data)))
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        return text

    def score(self, ca, ea, pa):
        return True in [fuzzy_match(ca, ea) for ca in eval(ca)]


class MMLU(Dataset):
    def __init__(self, data_path='benchmark/mmlu', result_path='result/mmlu', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'The following are multiple choice questions (with answers) about {subject}.\n\n{demos}\n\n{question}'
        self.question_template = "Q: {q}\nA) {a1}\nB) {a2}\nC) {a3}\nD) {a4}\nA: {a}"
        self.score_name = 'em'

    def reformat_data(self):
        subjects = [os.path.split(file[:-9])[1] for file in walk(self.data_path + '/data/test')]
        inputs = {'Source': [], 'ID': [], 'Input': [], 'CA': []}
        for subject in subjects:
            train_instances = pd.read_csv(self.data_path + f"/data/dev/{subject}_dev.csv", header=None)
            test_instances = pd.read_csv(self.data_path + f"/data/test/{subject}_test.csv", header=None)
            train_instances.columns = test_instances.columns = ['q', 'a1', 'a2', 'a3', 'a4', 'a']
            test_instances['ID'] = range(len(test_instances))
            test_instances['ID'] = test_instances['ID'].apply(lambda x: f'{subject}_{x}').tolist()
            train_instances = train_instances.to_dict(orient='records')
            test_instances = test_instances.to_dict(orient='records')
            if self.sample_num > 0:
                test_instances = random.sample(test_instances, k=self.sample_num)
            demos = [self.question_template.format(**ti) for ti in train_instances]
            demos = '\n'.join(demos)
            for test_instance in test_instances:
                inputs['Source'].append(self.data_path + f"/data/test/{subject}_test.csv")
                inputs['ID'].append(test_instance['ID'])
                inputs['CA'].append(test_instance['a'])
                test_instance['a'] = ''
                question = self.question_template.format(**test_instance).strip()
                input = self.prompt.format(**{'subject': subject, 'demos': demos, 'question': question})
                inputs['Input'].append(input)
        inputs = pd.DataFrame(inputs)
        inputs.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # check: A.
            r = re.search(r'([ABCD])\.', text)
            if r:
                return r.group(1)
            # check (A)
            r = re.search(r'\(([ABCD])\)', text)
            if r:
                return r.group(1)
            # check: A:
            r = re.search(r'([ABCD]):', text)
            if r:
                return r.group(1)
            # check A
            r = re.search(r'([ABCD])', text)
            if r:
                return r.group(1)
        return answer

    def score(self, ca, ea, pa):
        return exact_match(ca, ea)


class BBH(Dataset):
    def __init__(self, data_path='benchmark/bbh', result_path='result/bbh', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = '{demo}\n\nQ: {question}\nA: '
        self.score_name = 'em'

    def reformat_data(self):
        tasks = [os.path.split(filename)[1][8:-4] for filename in walk(self.data_path + '/data')]
        inputs = []
        for task in tasks:
            # 读取
            task_data_path = self.data_path + f'/data/samples_{task}.csv'
            instances = pd.read_csv(task_data_path)
            demo = read_file(self.data_path + f'/prompt/{task}.txt').split('-----')[1].strip()
            # 格式
            task_inputs = [[task_data_path, f'{task}_{idx}', self.prompt.format(demo=demo, question=instance['Question']),
                            instance['Answer']] for idx, instance in instances.iterrows()]
            # 采样
            if self.sample_num >= 1:
                task_inputs = random.sample(task_inputs, k=self.sample_num)
            elif 0 < self.sample_num < 1:
                task_inputs = random.sample(task_inputs, k=int(self.sample_num * len(task_inputs)))
            inputs.extend(task_inputs)
        inputs = pd.DataFrame(inputs, columns=['Source', 'ID', 'Input', 'CA'])
        inputs.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            if 'So the answer is ' in text:
                answer = text.split('So the answer is ')[1]
            else:
                answer = text.split('\n')[-1]
        return answer

    def score(self, ca, ea, pa):
        return ca in ea


class WinoGrande(Dataset):
    def __init__(self, data_path='benchmark/winogrand', result_path='result/winogrand', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'Choose the option that fill in the blank best.\n{sentence}\nA) {a1}\nB) {a2}\nAnswer:'
        self.score_name = 'em'

    def reformat_data(self):
        data = pd.read_csv(self.data_path + '/data/validation.csv', index_col=0)
        inputs = data.apply(lambda x: self.prompt.format(sentence=x['sentence'], a1=x['option1'], a2=x['option2']),
                            axis=1).tolist()
        answers = [['A', 'B'][a - 1] for a in data['answer'].tolist()]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        if self.sample_num > 0:
            print(f'Sample {self.sample_num} instances from {len(data)} instances!')
            data = data.sample(self.sample_num)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # check A)
            r = re.search(r'([AB])\)', text)
            if r:
                return r.group(1)
            # check: A.
            r = re.search(r'([AB])\.', text)
            if r:
                return r.group(1)
            # check: A:
            r = re.search(r'([AB]):', text)
            if r:
                return r.group(1)
            # check A
            r = re.search(r'([AB])', text)
            if r:
                return r.group(1)
        return answer

    def score(self, ca, ea, pa):
        return exact_match(ca, ea)


class Race(Dataset):
    def __init__(self, data_path='benchmark/race', result_path='result/race', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'The following are question (with answers) about reading comprehension.\n\nPassage: {passage}\nQuestion: {question}\nA) {a1}\nB) {a2}\nC) {a3}\nD) {a4}\nPlease answer with the letter of the correct answer.\nAnswer: '
        self.score_name = 'em'

    def reformat_data(self):
        data = read_file(self.data_path + '/data/samples_h.jsonl')
        inputs = [self.prompt.format(passage=d['passage'], question=d['question'], a1=d['options'][0], a2=d['options'][1],
                                a3=d['options'][2], a4=d['options'][3]) for d in data]
        answers = [['A', 'B', 'C', 'D'][d['ideal']] for d in data]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        # 采样
        if self.sample_num >= 1:
            k = self.sample_num
        elif 0 < self.sample_num < 1:
            k = int(self.sample_num * len(data))
        if self.sample_num > 0:
            print(f'Sample {k} instances from {len(data)} instances!')
            data = data.sample(k)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # check: A.
            r = re.search(r'([ABCD])\.', text)
            if r:
                return r.group(1)
            # check A)
            r = re.search(r'([ABCD])\)', text)
            if r:
                return r.group(1)
            # check: A:
            r = re.search(r'([ABCD]):', text)
            if r:
                return r.group(1)
            # check A
            r = re.search(r'([ABCD])', text)
            if r:
                return r.group(1)
        return answer

    def score(self, ca, ea, pa):
        return exact_match(ca, ea)


class Drop(Dataset):
    def __init__(self, data_path='benchmark/drop', result_path='result/drop', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'The following are question (with answers) about reading comprehension.\n\n{demo}\n\nPassage: {p}\nQuestion: {q}\nAnswer:'
        self.question_template = 'Passage: {p}\nQuestion: {q}\nAnswer: {a}'
        self.score_name = 'f1'

    def reformat_data(self):
        data = read_file(self.data_path + '/data/samples.jsonl')
        samples = read_file(self.data_path + '/data/fewshot.jsonl')
        demo = '\n\n'.join(
            [self.question_template.format(p=sample['passage'], q=sample['question'], a=sample['ideal'][0]) for sample in
             samples])
        inputs = [self.prompt.format(demo=demo, p=d['passage'], q=d['question']) for d in data]
        answers = [d['ideal'] for d in data]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        if self.sample_num >= 1:
            k = self.sample_num
        elif 0 < self.sample_num < 1:
            k = int(self.sample_num * len(data))
        if self.sample_num > 0:
            print(f'Sample {k} instances from {len(data)} instances!')
            data = data.sample(k)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = text

        patterns = [('[Aa]nswer:(.*?)(\.|,|\n|$)(\D|$)', 0),
                    ('[Aa]nswer.*?(:|is)[ \n](.*?)(\.|,|\n|$)(\D|$)', 1)]
        for pattern in patterns:
            r = re.findall(pattern[0], text)
            if len(r):
                answer = r[-1][pattern[1]]
                break

        while ':' in answer:
            answer = answer.split(':')[-1]
        while '(' in answer:
            answer = answer.split('(')[0]

        return answer.strip()

    def score(self, ca, ea, pa):
        ca = eval(ca)

        def list_in(lst, text):
            for e in lst:
                if e in text:
                    return 1
            return 0

        def _f1_score(prediction: str, ground_truth: str):
            prediction_tokens = normalize(prediction).split()
            ground_truth_tokens = normalize(ground_truth).split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        fscore = max([_f1_score(ea, answer) for answer in ca])

        if fscore == 0 and list_in(ca, pa):
            fscore = 1

        return fscore


class GSM8K(Dataset):
    def __init__(self, data_path='benchmark/gsm8k', result_path='result/gsm8k', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'Follow the given examples and answer the question.\n\n{demo}\n\nQ: {q}\nA: Let\'s think step by step'
        self.question_template = 'Q: {q}\nA: {a}'
        self.score_name = 'em'

    def reformat_data(self):
        data = read_file(self.data_path + '/data/samples.jsonl')
        samples = read_file(self.data_path + '/data/fewshot.jsonl')
        demo = '\n\n'.join(
            [self.question_template.format(q=sample['question'].strip(), a=sample['answer'].strip()) for sample in samples])
        inputs = [self.prompt.format(demo=demo, q=d["question"].strip()) for d in data]
        answers = [d['answer'] for d in data]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        # 采样
        if self.sample_num >= 1:
            k = self.sample_num
        elif 0 < self.sample_num < 1:
            k = int(self.sample_num * len(data))
        if self.sample_num > 0:
            print(f'Sample {k} instances from {len(data)} instances!')
            data = data.sample(k)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # So the answer is 123.
            r = re.search(r'So the answer is.*?(\d+)', text)
            if r:
                return r.group(1).strip()
            # answer is 123.
            r = re.search(r'answer is.*?(\d+)', text)
            if r:
                return r.group(1).strip()
            # 123
            r = re.findall(r'(\d+)', text)
            if len(r):
                return r[-1]
        return answer

    def score(self, ca, ea, pa):
        return ca.split('####')[1].strip() == ea


class Math(Dataset):
    def __init__(self, data_path='benchmark/math', result_path='result/math', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'Follow the given examples and answer the question. Highlight final answer with \\box{}.\n\n{demo}\n\nQ: {q}\nA: Let\'s think step by step'
        self.question_template = 'Q: {q}\nA: {a}'
        self.score_name = 'em'

    def reformat_data(self):
        data = read_file(self.data_path + '/data/samples.jsonl')
        samples = read_file(self.data_path + '/data/fewshot.jsonl')
        demo = '\n\n'.join(
            [self.question_template.format(q=sample['question'].strip(), a=sample['answer'].strip()) for sample in samples])
        inputs = [self.prompt.format('{}', demo=demo, q=d["question"].strip()) for d in data]
        answers = [d['answer'] for d in data]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        if self.sample_num >= 1:
            k = self.sample_num
        elif 0 < self.sample_num < 1:
            k = int(self.sample_num * len(data))
        if self.sample_num > 0:
            print(f'Sample {k} instances from {len(data)} instances!')
            data = data.sample(k)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            answer = last_boxed_only_string(text)
            answer = remove_boxed(answer)
        return answer

    def score(self, ca, ea, pa):
        return is_equiv(ea, ca)


class TruthfulQA(Dataset):
    def __init__(self, data_path='benchmark/truthfulqa', result_path='result/truthfulqa', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'Answer the following multiple choice questions.\n\n{demo}\n\n{q}'
        self.question_template = 'Q: {q}\nA: {a}'
        self.score_name = 'em'

    def reformat_data(self):
        data = read_file(self.data_path + '/data/samples.jsonl')
        samples = read_file(self.data_path + '/data/fewshot.jsonl')

        def number_to_letter(number):
            number = int(number)
            if 1 <= number <= 26:
                letter = chr(number + 64)  # 65 - 1 = 64
                return letter
            else:
                return "Number out of range (1-26)"

        def instance2str(instance):
            s = instance['question']
            for cidx, choice in enumerate(instance['choices']):
                s += '\n{}) {}'.format(number_to_letter(cidx + 1), choice)
            return s, number_to_letter(instance['answer'] + 1)

        demo = '\n\n'.join(
            [self.question_template.format(q=instance2str(sample)[0], a=instance2str(sample)[1]) for sample in samples])
        inputs = [self.prompt.format(demo=demo, q=f'Q: {instance2str(d)[0].strip()}\nA:') for d in data]
        answers = [instance2str(d)[1] for d in data]
        ids = list(range(len(data)))
        data = pd.DataFrame({'ID': ids, 'Input': inputs, 'CA': answers})
        if self.sample_num >= 1:
            k = self.sample_num
        elif 0 < self.sample_num < 1:
            k = int(self.sample_num * len(data))
        if self.sample_num > 0:
            print(f'Sample {k} instances from {len(data)} instances!')
            data = data.sample(k)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # check: A: A)
            r = re.findall(r'A: ([ABCDEFGHIJKLMN])', text)
            if len(r):
                return r[-1]
            # check: A.
            r = re.findall(r'([ABCDEFGHIJKLMN])\.', text)
            if len(r):
                return r[-1]
            # check A)
            r = re.findall(r'([ABCDEFGHIJKLMN])\)', text)
            if len(r):
                return r[-1]
            # check: A:
            r = re.findall(r'([ABCDEFGHIJKLMN]):', text)
            if len(r):
                return r[-1]
            # check A
            r = re.findall(r'([ABCDEFGHIJKLMN])', text)
            if len(r):
                return r[-1]
        return answer

    def score(self, ca, ea, pa):
        return exact_match(ca, ea)


class SESI(Dataset):
    def __init__(self, data_path='benchmark/sesi', result_path='result/sesi', sample_num=-1):
        super().__init__(data_path, result_path, sample_num)
        self.prompt = 'In each of the following statements, a situation is described followed by four comments that seem to offer alternative explanations. You are asked to choose the letter that corresponds to the one statement which in your judgment is the most appropriate, intelligent, or logical comment upon it.\n{situation}\n{question}\nA. {o1}\nB. {o2}\nC. {o3}\nD. {o4}'
        self.score_name = 'em'

    def reformat_data(self):
        data = pd.read_csv(self.data_path + '/data/data.csv')
        data['Input'] = data.apply(lambda x: self.prompt.format(situation=x['Situation'],
                                                                question=x['Question'],
                                                                o1=x['Option_0'],
                                                                o2=x['Option_1'],
                                                                o3=x['Option_2'],
                                                                o4=x['Option_3']), axis=1).tolist()
        data = data[['Type', 'ID', 'Input', 'Answer']]
        data.rename(columns={'Answer': 'CA'}, inplace=True)
        if self.sample_num > 0:
            print(f'Sample {self.sample_num} instances from {len(data)} instances!')
            data = data.sample(self.sample_num)
        data.to_csv(self.input_path, index=False)

    def extract_answer(self, text):
        answer = 'None'
        if type(text) is str:
            text = text.strip()
            # check: A.
            r = re.search(r'([ABCD])\.', text)
            if r:
                return r.group(1)
            # check A)
            r = re.search(r'([ABCD])\)', text)
            if r:
                return r.group(1)
            # check: A:
            r = re.search(r'([ABCD]):', text)
            if r:
                return r.group(1)
            # check A
            r = re.search(r'([ABCD])', text)
            if r:
                return r.group(1)
        return answer

    def score(self, ca, ea, pa):
        ca = ['A', 'B', 'C', 'D'][ca]
        return exact_match(ca, ea)