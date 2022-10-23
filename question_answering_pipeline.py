from passage_classify.PassageClassify import PassageClassifier
from sentence_detect.SentenceDetect import SentenceDetectionModel

file_name = "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Gujarat.txt"

passage_classifier = PassageClassifier(file_name)

sentence_detection = SentenceDetectionModel(10, None, "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/Project/NaturalLanguageProcessing_11411-11611/sentence_detect/Model_addattention_3_hidden_layer_8_epochs_5e-6_dict_Try1.pt")

"""
questions = ["What is the capital of Meghalaya?","What is the official language of Meghalaya?",
"Is the Siju Wildlife Sanctuary located in Meghalaya?", "Where is the largest variety of orchids found in Meghalaya?","Is the rural sex ratio higher than the urban female sex ratio in Meghalaya?", "How many states in India have a Christian majority?","Which is the second most spoken language in Meghalaya?","Which state has the wettest region in India?",
"On how many sides is Meghalaya surrounded by Bangladesh?",
"Does Meghalaya follow a patrilineal system of lineage and inheritance?",
"When did the Assam Reorganisation (Meghalaya) Act come into effect?",
"What is the height of the Shillong Peak?",
"Is the large Indian parakeet the largest bird in Meghalaya?",
"Do Jaintias form the largest tribal group in Meghalaya?",
"Do Muslims form the largest minority in Meghalaya?",
"How many Shakti peethas exist in the world?",
"How many languages are used as a medium for instruction in schools in Meghalaya?",
"How many autonomous district councils are present in Meghalaya?",
"Does Meghalaya generate excess electricity?",
"Is Meghalaya the most literate state in India?",
"How far is the Nongkhnum Island from Nongstoin?",
"Which three rivers surround the Nongkhnum Island?"]
"""
"""
questions = ["Where is the Sagittarius constellation located?",
"Is Sigma Sagittarii the brightest star in the Sagittarius constellation?",
"How far from the Earth is the Omega Nebula located?",
"Who discovered the Trifid Nebula?",
"What is the diameter of the Trifid Nebula?",
"How far from the Earth is the Trifid Nebula located?",
"What is the diameter of the NGC 6445 nebula?"]
"""

questions = [ "Is Aretha Franklin one of Gujarat Notable people?", 
"What ethnic group is indigenous to Gujarat?",
"What leader preceded Skandagupta?",
"What was India's first government?",
"Did the ruler of the Kshatraps  (100 CE) found the Kardamaka dynasty?",
"Was Chatrapati Shivaji a Maratha ruler?",
"Selling alcohol is not allowed in Gujarat?",
"What is the second largest river in Gujarat?",
"Was Shah Jahan the Subahdar of Gujarat?",
"Is Ahmedabad Railway Station located in Paris?",
"What year did the Mughal emperor Akbar conquer and annex the Sultanate of Gujarat to the Mughal Empire?",
"Which ruler of Girinagar built a dam on the Sudarshan lake?",
"Does Gujarat have a high Jewish population?",
"Does Gujarat contain one of the largest and most prominent archaeological sites in India?",
"Where can you find Asiatic lions in the wild?",
"In which present day Indian state was the ancient city of Lothal located?",
"Who attempted and failed to conquer Gujarat and annex it to his empire in 1197?",
"Is Gujarat the tenth most populous state?",
"Is the name of Gujarat derived from a dynasty from over 10000 years ago?",
"Is the Gujarati food primarily pescatarian?",
"What did the Zoroastrian refugees  become known as?",
"Does Gujarat have what is considered to be one of the oldest seaports?",
"Was Qutbuddin Aibak successful in his attempt to conquer Gujarat?",
"Is the economy of Gujarat the largest in India?",
"In which present day Indian state was the ancient city of Dholavira located?",
"Does Gujarat rank the 1st among Indian states and union territories in human development index?",
"In which century did the Gupta empire begin to decline?",
"What was the imperial title of the sixth Mughal Emperor ruling with an iron fist over most of the Indian subcontinent?",
"Was the city of Surat once considered one the most import ports in the world?",
"What is the first Gujarati film that was released?",
"When was confusion over whether Junagadh would join India or Pakistan resolved?",
"Does Gujurat have a GDP of more than 250 Billion USD?",
"In which state was India's first port established?",
"Was Gujarat one of the main central areas of the Indus Valley Civilization?",
"Is there evidence of trade between Gujarat and the United States going back 3000 years?",
"Is Bengali Gujarat's official language?",
"What is desert in Gujarati?",
"Was Gujarat one of the original 12 subahs?",
"Does Gujarat have the lowest sex ratio among the 29 states in India?",
"What is the largest city of Gujarat?",
"How many states in India prohibit the sale of alcohol?",
"What is the name of the second longest river in Gujarat?",
"What is the third largest religion in Gujurat?",
"In which present day Indian state was the ancient city of Gola Dhoro located?",
"Who ruled Girinagar from 322 BCE to 294 BCE?",
"Which Portugese explorer described Surat in 1514?",
"In what museum can you see the skeleton of a whale?",
"Which empire defeated the Muslim Mughals in the 17th century?",
"Does Gujarat encompass more number of Indus Valley Civilzation sites than any other state?",
"Is Gujarat the largest state in India by area?",
"Is the capital city of Gujarat the same as the most populous city?",
"Is Chongqing growing faster than Ahmedabad in 2010's Forbes' list of the world's fastest growing cities?",
"How many districts does Gujarat have?",
"Does Gujurat have more men or women?",
"Is the ancient city of Dholavira a small site in India?",
"Which party won in the 1995 Assembly elections?",
"Did the ruler of the Saka satraps (100 CE) found the Kardamaka dynasty?",
"Were is the Indus Valley civilisation centered at?",
"When was the first Gujarati film, Narsinh Mehta, released?",
"How much area in square kilometres does Gujarat cover?",
"What is the population of Gujarat?",
"Were the inhabitants of ancient Gujarat commercialy active?",
"Who was the largest producer of milk in the  world in 2010?",
"How long was Narendra Modi the chief minister of Gujarat?",
"Where was Aurangzeb born?",
"The Gujarat encomprasses 21 sites of the ancient Indus Valley civilisation?",
"Who was the kshatrapa dynasty replaced by?",
"What are Gujarat's major cities?",
"What dynasty is Gujarat derived from?",
"Is Hindi the second language in Gujarat in 2001?",
"Who was the third son and sixth child of Shah Jahan and Mumtaz Mahal?",
"What is the fifth-largest Indian state by area?",
"What is the population of Gujarat?",
"Gujarat was also known as Pratichya and Varuna?"]

related_sentences = []
for question in questions:
    question_related_sentences = []
    passages = passage_classifier.get_related_passages(question)
    #for passage in passages[0]:
    sentences = sentence_detection.get_predicted_sentences(passages[0], question)
    question_related_sentences.extend(sentences)
    print(question)
    print(question_related_sentences)
    print()
    related_sentences.append(question_related_sentences)
    
