import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name)

    def _construct_prompt(
        self,
        question: str,
        relevant_sentences: list[str]
    ) -> str:
            
        context = str(" ".join(relevant_sentences))

        return f'question: {question} context:{context}'


    def answer_question(
        self,
        question: str,
        relevant_sentences: list[str]
    ) -> str:
        input = self._construct_prompt(question, relevant_sentences)

        encoded_input = self.tokenizer([input],
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)
        output = self.model.generate(input_ids = encoded_input.input_ids,
                                    attention_mask = encoded_input.attention_mask)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output

    def answer_questions(
        self,
        questions: list[str],
        relevant_sentences: list[list[str]]
    ):

        if len(questions) != len(relevant_sentences):
            raise ValueError(
                "Arguments questions and relevant_sentences must have the same length"
            )

        return [
            self.answer_question(
                questions[i], relevant_sentences[i]
            ) for i in range(len(questions))
        ]