from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import torch.nn.functional as F
from transformers.utils import logging

class T5AnswerGenerator():
    def __init__(self):
        self.model_name = "mzhou08/t5-base-finetuned-context-dataset"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
    def _construct_prompt(
        self,
        question,
        relevant_sentences
    ) -> str:
            
        context = str(" ".join(relevant_sentences))

        return f'question: {question} context:{context}'


    def answer_question(
        self,
        question,
        relevant_sentences
    ) -> str:
        input = self._construct_prompt(question, relevant_sentences)

        encoded_input = self.tokenizer([input],
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)
        output = self.model.generate(input_ids = encoded_input.input_ids,
                                    attention_mask = encoded_input.attention_mask, max_new_tokens=256)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output

    def answer_questions(
        self,
        questions,
        relevant_sentences
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