from document_preprocess import *
from document_classify import *

documents = DocumentBase("documents").read()
doc = documents["documents/Spanish_language.txt"]
DocumentPreprocessor.decompose(doc)
