import torch
from sentence_transformers import SentenceTransformer, util
import stanza

LINES_PER_PASSAGE = 10;


class PassageClassifier(torch.nn.Module):
    def __init__(self, file_name):
        super(PassageClassifier, self).__init__()
        
        stanza.download('en')
        self.tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
        self.sentences = self.tokenize_file(self.read_and_clean_file(file_name))
        
        
        self.model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')#.to(torch.device("cuda:0"), dtype=torch.float, non_blocking=True)
        self.sentence_embeddings = [torch.tensor(self.model.encode(sentence)) for sentence in self.sentences]
        print(f"Generated embeddings for {len(self.sentences)} sentences")
    
    def classify(self, question):
        question_embedding = torch.tensor(self.model.encode(question))
        dot_scores = sorted({i:util.dot_score(question_embedding, 
                                   self.sentence_embeddings[i])[0] for i in range(len(self.sentence_embeddings))}.items(), key=lambda x:x[1], reverse=True)
        print(dot_scores[0])
        return dot_scores
    
    def read_and_clean_file(self, file_name):
        clean_text = open(file_name).read().replace('\n', '')
        clean_text = clean_text.replace('=', '')
        return clean_text
    
    def tokenize_file(self, clean_text):
        stanza_result = self.tokenizer(clean_text)
        sentences = [stanza_result.sentences[i].text for i in range(len(stanza_result.sentences))]
        return sentences
    
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
        result = []
        start, end = self.find_passage_start_and_end(sentence_number[0][0], len(self.sentences))
        result.append("\n".join(self.sentences[start: end]))
        start, end = self.find_passage_start_and_end(sentence_number[1][0], len(self.sentences))
        result.append("\n".join(self.sentences[start: end])) 
        return result
    

