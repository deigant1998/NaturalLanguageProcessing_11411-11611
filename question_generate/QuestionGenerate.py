
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

class QuestionGenerator():
    def __init__(self):
        self.model_name = "Deigant/t5-base-finetuned-qg-hard-medium"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        
    def create_prompt(self, related_sentences) -> str:
        return f'generateQuestion: {related_sentences}'
    
    def generate_question(self, context):
        prompt = self.create_prompt(context)
        encoded_input = self.tokenizer([prompt],
                                    return_tensors='pt',
                                    max_length=1024,
                                    truncation=True)
        output = self.model.generate(input_ids = encoded_input.input_ids,
                                    attention_mask = encoded_input.attention_mask, 
                                    num_beams = 2, do_sample = True, max_length=128, num_return_sequences = 1,
                                    repetition_penalty = 1.2)
        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output;
        
        
        