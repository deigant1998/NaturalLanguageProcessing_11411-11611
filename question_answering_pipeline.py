from passage_classify.PassageClassify import PassageClassifier
from sentence_detect.SentenceDetect import SentenceDetectionModel
from answer_generate.AnswerGenerate import T5AnswerGenerator

file_name = "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/indian_states/Gujarat.txt"

passage_classifier = PassageClassifier(file_name)
sentence_detection = SentenceDetectionModel(10, None, "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/Project/NaturalLanguageProcessing_11411-11611/sentence_detect/Model_doubleattention_3_hidden_layer_8_epochs_5e-6_dict_Try1.pt")
answer_generator = T5AnswerGenerator()

questions = ["What religion is the most popular in Gujarat according to the 2011 census?",
"Is London a major city in Gujarat ?",
"Who ruled Gujarat in the 8th and 9th centuries CE?",
"Is over half of Gujarat's geographical area used for agriculture?",
"Is Gujarat one of four Indian states to prohibit the sale of alcohol?",
"Which ruler of modern-day Junagadh built a dam on the Sudarshan lake?",
"How long is the coastline of Gujarat?",
"Does Gujarat contain the longest coastline in India?",
"What is the capital city of Gujarat?",
"Is Gujarat landlocked?",
"Is Gujarat bounded by Europe to the southwest?",
"Gujarat is derived from which dynasty?",
"Is Gujarat bounded by the Arabian Sea to the northwest?",
"Who ordered his edicts engraved in the rock at Girinagar?",
"Was Surat a thriving and luxurious place for some in the 1500s?",
"When does the largest number of sanghas visit Ambaji village?",
"What are three important sites of the ancient Indus River Valley civilization?",
"Was Aurangzeb the sixth Mughal Emperor?",
"Does Cambay have beautiful architecture?",
"Who ruled Gujarat in the 8th and 9th centuries CE?",
"How many population does Gujarat have?",
"What is the most common language spoken among Gujarati Muslims?",
"What is one of the dinosaur species found at Balasinor?",
"Which state encompasses the highest number of ancient Indus Valley Civilisation sights?",
"In which year was the first successful agitation that ousted an elected government after the Independence of India ?",
"Who founded Physical Research Laboratory and envisioned Indian Institute of Management Ahmedabad?",
"Who is Asia's biggest dairy?",
"Did one of Jehangir's grandchildren become a Mughal Emperor?",
"Is Gujarat the largest Indian state by  population?",
"Is Gujuart the largest Indian state by are?",
"Which country administered enclaves along the Gujarati coast, including Daman and Diu for over 450 years?",
"What is the official language of Gujarat?",
"What is the third most practiced religion in Gujarat?",
"What is the official language of Gujarat?",
"Where was India's first port established?",
"Are there Asiatic Lions in Gujurat?",
"What are the three main sources of growth in Gujarat's agriculture?",
"Whose grandson ordered his edicts engraved in the rock at Girinagar?",
"Is rap a Gujarati folk music?",
"Who ruled modern-day Junagadh from 322 BCE to 294 BCE?",
"Does The state of Bihar allow the sale of alcohol?", 
"Did Pushyagupta live over 2000 years ago?",
"Does Gujarat come under the northen Railway Zone of the Indian Railways?",
"Is the name of Gujarat derived from a dynasty from over 1000 years ago?",
"Where was India's first port established?",
"Where was Amul formed?",
"How many members are in Gujarat's  Legislative Assembly?",
"Does The state of Mizoram allow the sale of alcohol?", 
"Did Gujarat once have one the most important ports in the world?",
"Which country was the first European power to arrive in Gujarat?",
"What was the capital when the Chaulukya dynasty ruled Gujarat?",
"Is Gujarat the largest Indian state by  area?",
"Who is the father of Aurangzeb?",
"What is Gujarat's official language?",
"What is the second most common religion in Gujarat?",
"What dynasty was replaced by the Gupta empire with the conquest of Gujarat by Chandragupta Vikramaditya?",
"Was Gijarat known by the ancient Greeks?",
"What percent of the population in Gujarat natively speaks Gujarati?",
"Did the ruler of the Western Satraps (100 CE) found the Kardamaka dynasty?",
"Is it prohited to sell alcohol in Gujarat?",
"In which year did the British East India Company form their first base in India?",
"Is Gujarat known for its manufacturing?",
"Based on the 2001 census, what is the second most spoken language in Gujarat?",
"Who was Emperor Ashoka the Great's grandfather?",
"Whose grandson ordered his edicts engraved in the rock at Junagadh?",
"What is Gujarat's population rounded to the closest million in 2011?",
"What is the tallest tower in Gujarat?",
"How long did the government following the INC stay in power?",
"Is Gujarat on the west coast of India?",
"Is there any clue of medieval trade in the western Indian Ocean?",
"What is the rank of Gujarat among Indian states and union territories in human development index?",
"Had Gujarati merchents in the 16th century earned an international reputation for commerical acumen?",
"What encouraged the visit of merchants from areas like Cairo in the early 16th century?",
"Is the only wild population of the Asiatic lion in the world in Gujarat the Lion Forest National Park ?",
"Is Gandhunagar the capital and largest city of Gujarat?",
"Did the Gurjara-Pratihara dynasty rule in the 6th century CE?",
"Is the White Revolution of India the world's biggest dairy development program?",
"Did the Kshatrapa dynasty come before the Gupta Empire?",
"Does Gujarat have the best pharmaceutical industry in India?",
"Is it illegal to sell alcohol in Gujarat?",
"What movement was a students‚Äô and middle-class people‚Äôs movement against economic crisis and corruption in public life?",
"Which Indian state is described in the story of a merchant of King Gondophares and Apostle Thomas in 16th century manuscripts?",
"Is Ahmedabad Gujarat's capital city?",
"Which individual was defeated and overthrown by the forces of Alauddin Khalii from Delhi in 1297?",
"Which city did Sultan Ahmed Shah name as capital of Gujarat?",
"In which peninsula does most of Gujarat's coastline lie?",
"What is the largest river in Gujarat?",
"In which year did India become independent?",
"What were the three major Indian  dynasties?",
"Who was the successor of Qasim?",
"Was Ahmedabad on the top of Forbes' list of the world's fastest growing cities in 2010?",
"How much area in square miles does Gujarat cover?",
"How many districts and talukas does  Gujarat have?",
"Is Dholavira the largest site of the ancient Indus Valeey civilisation?",
"Who was the ruler of Kshatraps in 100CE?",
"What is the population of Gujarat according to the 2011 census data?",
"What is the name of the women's costumes for raas-garba?",
"Does Gujarat have the shortest sea in India?",
"Is Gujarat located in the North of India?",
"Is it illegal to sell alcohol in Punjab, India?",
"Is there evidence of trade between Gujarat and Egypt going back 3000 years?",
"How many original subahs were established by Mughal Emperor Akbar?",
"How many states prohibit alcohol in India?",
"Did Jehangir have more than 5 grandchildren?",
"Did Pushyagupta rule modern-day Junagadh?",
"Is pop a Gujarati folk music?",
"Where is the capital city of Gujarat?",
"Did Pushyagupta build a dam on the Sudarshan lake?",
"Did the ancient city of Dholavira belong to the Indus Valley civilisation?",
"Does Gujarat border Madhya Pradesh to the north?",
"Is Gujarati the second language in Gurjart in 2001?",
"What was Gujarat's major religion in 2011?",
"What is the name of the oldest seaport in Gujarat?",
"What was Gujarat's second major religion in 2011 ?",
"Who was the ruler of Western Satraps in 100CE?",
"Is Gujurat's capital city the same as it's capital?",
"Is Aretha Franklin one of Gujarat Notable people?", 
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

answers = []

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
    answer = answer_generator.answer_question(question, question_related_sentences)
    print(answer)
    answers.append(answer)
    
