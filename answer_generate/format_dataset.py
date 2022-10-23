import pandas as pd

def create_qa_dataset(data_path: str) \
-> tuple[list[str], list[list[str]]]:

    question_data = pd.read_excel(data_path)

    questions = list(question_data["question"])

    all_relevant_sentences = []

    for idx, row in question_data.iterrows():
        relevant_idxs = [int(x) for x in str(row["Valid Sentences"]).split(",")]
        sentences = row["Passage (10 sentences)"].split("\n")

        relevant_sentences = [sentences[i] for i in relevant_idxs]

        all_relevant_sentences.append(relevant_sentences)

    return (questions, all_relevant_sentences)