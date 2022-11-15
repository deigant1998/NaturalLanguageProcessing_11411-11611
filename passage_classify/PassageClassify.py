import torch
from sentence_transformers import SentenceTransformer, util
from utils import FileRead
import stanza

LINES_PER_PASSAGE = 10;


class PassageClassifier(torch.nn.Module):
    def __init__(self, file_name):
        super(PassageClassifier, self).__init__()
        stanza.download('en', verbose=False)
        self.tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
        self.file_reader = FileRead.FileReader()
        self.sentences = self.file_reader.get_sentences(file_name)
        self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')#.to(torch.device("cuda:0"), dtype=torch.float, non_blocking=True)
        self.sentence_embeddings = self.model.encode(self.sentences, convert_to_tensor=True)
        
    def classify(self, question):
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        hit = util.semantic_search(question_embedding, self.sentence_embeddings, top_k=1)[0]
        
        #dot_scores = util.dot_score(question_embedding, self.sentence_embeddings)[0].cpu().tolist()
        
        #dot_scores = sorted({i:dot_scores[i] for i in range(len(self.sentence_embeddings))}.items(), key=lambda x:x[1], reverse=True)
        return hit[0]['corpus_id']
    
    def find_passage_start_and_end(self, sentence_index, len_sentences):
        start = sentence_index - 5
        end = start + 10
        
        if(start < 0):
            start = 0
            end = 10
        elif (end > len_sentences):
            end = len_sentences
            start = len_sentences - 10
        
        return start, end
    
    def get_related_passages(self, question):
        sentence_number = self.classify(question)
        start, end = self.find_passage_start_and_end(sentence_number, len(self.sentences))
        return self.sentences[start: end]
        #start, end = self.find_passage_start_and_end(sentence_number[1][0], len(self.sentences))
        #result.append("\n".join(self.sentences[start: end])) 
        #return result
    

