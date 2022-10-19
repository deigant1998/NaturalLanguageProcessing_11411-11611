import os
import torch
from transformers import BertModel, BertTokenizer

# Object representing the collection of documents that the document classifier will read from
class DocumentBase:
    def __init__(self, base_path):
        self.base_path = base_path

    def read(self):
        files = []
        for path, dirnames, filenames in os.walk(self.base_path):
            for f in filenames:
                files.append(os.path.join(path, f))

        documents = {}
        for f in files:
            fp = open(f, "r")
            documents[f] = fp.read()
            fp.close()

        return documents

def make_passages_from_document(document):
    return [passage for passage in document.split("\n") if 32 <= len(passage) and len(passage) < 768]

cosine_similarity = torch.nn.CosineSimilarity(dim=0)

class DocumentClassifier:
    def __init__(self, document_base):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model.eval()

        self.documents = document_base.read()
        self.passages = {(source, n):passage for source, document in self.documents.items() for n, passage in enumerate(make_passages_from_document(document))}
        self.passage_embeddings = {key:self.generate_embedding(passage) for key, passage in self.passages.items()}
        print(f"Generated embeddings for {len(self.passage_embeddings)} passages")

    def classify(self, question):
        question_embedding = self.generate_embedding(question)
        p = [(k, v) for k, v in self.passage_embeddings.items()]
        p.sort(key=lambda element: cosine_similarity(question_embedding, element[1]))
        print([k for k, v in p])
        for i in range(3):
            b = p[-1 - i]
            print(f"{i}-th best similarity is {b[0]}")
            print("Passage text:")
            print(self.passages[b[0]])

    def generate_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        token_vecs = hidden_states[-2][0]
        embedding = torch.mean(token_vecs, dim=0)
        return embedding
