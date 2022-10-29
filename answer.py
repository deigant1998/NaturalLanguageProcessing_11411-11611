#!/usr/bin/env python3
from passage_classify.PassageClassify import PassageClassifier
from sentence_detect.SentenceDetect import SentenceDetectionModel
from answer_generate.AnswerGenerate import T5AnswerGenerator
import sys
from transformers.utils import logging

file_name = "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Gujarat.txt"


if __name__ == "__main__":
    context_file_name = sys.argv[1]
    questions_file_name = sys.argv[2]
    logging.set_verbosity_error()
    logging.disable_progress_bar()
    passage_classifier = PassageClassifier(context_file_name)
    sentence_detection = SentenceDetectionModel(10, None, "sentence_detect/Model_doubleattention_3_hidden_layer_8_epochs_5e-6_dict_Try1.pt")
    answer_generator = T5AnswerGenerator()

    related_sentences = []
    answers = []
    
    with open(questions_file_name) as questions_file:
        questions = questions_file.readlines()
        for question in questions:
            question_related_sentences = []
            passage = passage_classifier.get_related_passages(question)
            sentences = sentence_detection.get_predicted_sentences(passage, question)
            question_related_sentences.extend(sentences)
            related_sentences.append(question_related_sentences)
            answer = answer_generator.answer_question(question, question_related_sentences)
            print(answer)
            answers.append(answer)

    
    
