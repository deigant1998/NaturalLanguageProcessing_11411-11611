from keybert import KeyBERT

class KeywordGenerate:
    def __init__(self):
        self.kw_model = KeyBERT()
        
    def get_keywords(self, sentence):
        keywords = self.kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1,1), diversity = 0.5, use_mmr=True, top_n=4)
        keywords = [keyword[0] for keyword in keywords]
        return keywords