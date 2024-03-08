# Situational Evaluation of Social Intelligence

- An implementation for paper "Academically intelligent LLMs are not necessarily socially intelligent".

## Requirements

General

- Python (verified on 3.8)

Python Packages

- see requirements.txt

```bash
conda create -n ecom python=3.8
conda activate ecom

pip install -r requirements.txt
```

## Quick Start

### Data Format

Before evaluation, you should first download data in SESI/data/data.csv.

The data format is as follows:

- ```ID``` refers to the id of the test item.
- ```Type``` refers to the social intelligence ability tested by the test item.
- ```Situation``` refers to the social situation.
- ```Question``` refers to the social question.
- ```Option``` refers to four comments that seem to offer alternative explanations.
- ```Answer``` refers to the index of the most appropriate, intelligent, or logical comment.

| ID      | Type    | Situation                                                    | Question                                                     | Option_0                                                     | Option_1                                                     | Option_2                                                     | Option_3                                                     | Answer |
| :------ | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------ |
| 16l2bxv | Empathy | A person's 6-year-old son found his dog dead  at his father's house, and the grandfather isn't giving any answers to the  child's inquiries about the dog's death. | Why is the grandfather not providing any  answers to the child about the dog's death? | A lack of answers and uncertainty about  explaining the dog's death to his 6-year-old has caused him to avoid  discussing it. | He is not providing any answers to the child  about the dog's death because he does not understand the importance of  addressing the child's questions and providing appropriate explanations. | He is possibly in shock or emotionally  overwhelmed by the sudden loss of their family pet, which is why he is not  providing any answers to the child about the dog's death. | He may be hiding something, like neglect or  possible foul play, which is why he is not providing any answers to the child  about the dog's death. | 3      |

### LLM Evaluation

Code for LLM evaluation can be found in folder evaluation.

```bash
python main.py \
	--dataset sesi \
	--model 'gpt-3.5-turbo-0613' \
	--sample_num -1 \
	--thread_num 8 \
	--chunk_size 5 \
	--seed 42 \
	--api_base '' \
	--api_key ''
```

