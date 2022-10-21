import pandas as pd

def create_qa_dataset(data_path: str) -> pd.DataFrame:
    question_data = pd.read_excel(data_path)
    for idx, row in question_data.iterrows():
        relevant_idxs = [int(x) for x in str(row["Valid Sentences"]).split(",")]
        sentences = row["Passage (10 sentences)"].split("\n")

        relevant_sentences = [sentences[i] for i in relevant_idxs]

        context = str(" ".join(relevant_sentences))
        
        question_data.at[idx, "context"] = context

        question_data.at[idx, "prompt"] = f'question: {row["question"]} context:{context}'

    return question_data