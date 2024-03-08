import json
import os

def read_file(file_name, split_str=None):
    if 'jsonl' in file_name:
        datas = []
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                datas.append(data)
        return datas
    elif 'json' in file_name:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    else:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = f.read()
        if split_str:
            elements = data.split(split_str)
            elements = [e.strip() for e in elements if e.strip()]
            return elements
        else:
            return data


def write_file(file_name, data, split_str=None):
    if type(data) is list:
        lists = data
        if 'jsonl' in file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    json.dump(element, f)
                    f.write('\n')
        else:
            split_str = '\n' if not split_str else split_str
            with open(file_name, 'w', encoding='utf-8') as f:
                for element in lists:
                    f.write(str(element))
                    f.write(split_str)
    elif type(data) is dict:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(data, f)
    else:
        with open(file_name, 'w', encoding='utf-8') as f:
            f.write(str(data))


def walk(path):
    filenames = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filename = os.path.join(root, file)
            filenames.append(filename)
    return filenames


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        # 如果不存在，则创建文件夹
        os.makedirs(folder_path)
        print(f"Create '{folder_path}'!")
