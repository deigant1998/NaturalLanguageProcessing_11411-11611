import pandas as pd
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelWithLMHead,
    pipeline
)
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

class T5AnswerGenerator():
    def __init__(self):
        model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelWithLMHead.from_pretrained(model_name)

    def answer_questions(self, dataset: pd.DataFrame):
        answers = []
        num_correct = 0
        num_incorrect = 0

        for idx, row in dataset.iterrows():
            input = row["prompt"]
            encoded_input = self.tokenizer([input],
                                        return_tensors='pt',
                                        max_length=512,
                                        truncation=True)
            output = self.model.generate(input_ids = encoded_input.input_ids,
                                        attention_mask = encoded_input.attention_mask)
            output = self.tokenizer.decode(output[0], skip_special_tokens=True)
            answers.append((input, output))
            if (str(output).lower().find(str(row["answer"]).lower()) != -1):
                print("Correct")
                num_correct += 1
                continue
            print("Incorrect Answer:")
            print(f'Difficulty: {row["difficulty"]}')
            print(input)
            print(output)
            print(f'correct answer: {row["answer"]}')
            num_incorrect += 1

            print(f'Correct: {num_correct} Incorrect: {num_incorrect} Total:{num_correct + num_incorrect}')