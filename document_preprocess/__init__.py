def is_heading(text):
    l = 2
    text = text.strip()
    return text[:l] == "==" and text[-l:] == "=="

class DocumentPreprocessor:
    @staticmethod
    def decompose(document_text):
        normalized_document_text = document_text.strip("\r") + "\n\n\n"
        sections = [sec for sec in document_text.split("\n\n\n") if len(sec) > 256]
        paragraphs = [para for sec in sections for para in sec.split("\n\n") if len(sec) > 8]
        lines = [line for para in paragraphs for line in para.split("\n") if not is_heading(line)]
        print(lines)
