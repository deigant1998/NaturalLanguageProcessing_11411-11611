import stanza 

class FileReader():
    def __init__(self):
        stanza.download('en', verbose=False)
        self.tokenizer = stanza.Pipeline(lang='en', processors='tokenize')
        
    def read_and_clean_file(self, file_name):
        clean_text = open(file_name).read().replace('\n', '')
        clean_text = clean_text.replace('=', '')
        return clean_text
    
    def tokenize_file(self, clean_text):
        stanza_result = self.tokenizer(clean_text)
        sentences = [stanza_result.sentences[i].text for i in range(len(stanza_result.sentences))]
        return sentences
    
    def get_sentences(self, file_name):
        return self.tokenize_file(self.read_and_clean_file(file_name))