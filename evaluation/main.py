import argparse
import openai

from eval import eval


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--dataset", default='nq', type=str,
                        choices=['nq', 'mmlu', 'bbh', 'winogrande', 'race', 'drop', 'gsm8k', 'math', 'truthfulqa', 'sesi'],
                        help="Benchmark.")
    parser.add_argument("--model", default='gpt-3.5-turbo-0613', type=str, help="Model.")
    parser.add_argument('--sample_num', type=int, default=-1, help="Number of test items.")
    parser.add_argument('--thread_num', type=int, default=8)
    parser.add_argument('--chunk_size', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--api_base", default='', type=str, help="API base for openai models.")
    parser.add_argument("--api_key", default='', type=str, help="API key for openai models.")

    args = parser.parse_args()

    if len(args.api_base):
        openai.api_base = args.api_base
    if len(args.api_key):
        openai.api_key = args.api_key

    # Eval
    eval(args.model, args.dataset, args.sample_num, args.thread_num, args.chunk_size, args.seed)


if __name__ == '__main__':
    main()