from venv import create
from format_dataset import *
from AnswerGenerate import *

qns, rel_sents = create_qa_dataset("../data/context-dataset.xlsx")

ans = T5AnswerGenerator().answer_questions(qns, rel_sents)

for i in range(len(qns)):
    print(
        f"""
        {qns[i]}\n
        {rel_sents[i]}\n
        ====\n
        {ans[i]}\n
        ====\n
        """
    )