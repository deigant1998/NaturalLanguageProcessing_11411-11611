from passage_classify.PassageClassify import PassageClassifier
from sentence_detect.SentenceDetect import SentenceDetectionModel
from answer_generate.AnswerGenerate import T5AnswerGenerator

file_name = "/Users/deigant/Desktop/CMU/CMU Course Materials/11-611/hw1/HW01/documents/constellations/leo_constellation.txt"

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

questions = ["Leo remains one of how many modern constellations?",
"How many stars is the sickle marked by?",
"Is the carbon star CW Leo (IRC +10216) is the brightest star in the night sky at the infrared N-band (10 Œºm wavelength)?",
"Is Leo one of the constellations of the zodiac?",
"Is gold one of the major elements found in the orbit of the two galaxies found within Leo?",
"What type of star is Iota Leonis?",
"NGC 2903 is similar to which galaxy?",
"Is the earth in the Leo constellation?",
"What animal is Leo?",
"What did Leo represent to ancient Greeks?",
"Which constellation is located between Cancer the crab to the west and Virgo the maiden to the east?",
"What is Leo Latin for?",
"Was leo killed by Gilgamesh?",
"Is Leo an astrological sign of August?",
"Does Leo only contain one galaxy?",
"Apart from hydrogen, what is the other element found within galaxies of Leo?",
"How many stars in the Leo constellation are in the first or second magnitude?",
"Is Leo the most recently recognized constellation?",
"Is Messier 65 part of Leo Triplet?",
"Are all the 88 modern constellations today as recognizable as Leo?",
"Is Alpha Leonis visible in binocular?",
"Does Leo contain Messier 65?",
"Who killed the monster Humbaba?",
"How many modern constellations are there?",
"Is Algieba more than 1260 light-years away from Earth?",
"Are there multiple types of Leonid showers?",
"What two months do Leonid showers occur?",
"Does Leo contain the galaxy Messier 98?",
"What is 7 meteors per hour the normal peak rate of the Leonidis in November?",
"How often does Comet Tempel-Tuttle outburst?",
"Is Regulus a black main-sequence star?",
"What shape does Leo take?",
"Is November 18 before the peak days of the Leonid Meteor shower that have a radiant close to Gamma Leonis?",
"Why did Heracles kill the Nemean Lion?",
"Is Regulus farther than 70 light-years from Earth?",
"What is the meaning of Leo in Latin?",
"Is Beta Leonis in the middle of the Leo constellation?",
"What language is Leo's name?",
"Does Beta Leonis means \"the little king\"?",
"What color is \"the little king\" star of Leo?",
"Is the Leo constellation hard to recognize?",
"Is Leo easily recognizable?",
"How many stars of first or second magnitude are there in Leo constellation?",
"When is the sun considered to be in the sign Leo?",
"The lion's tail is marked by which star?",
"What is the magnitude of 40 Leonis?",
"Which galaxies are part of the Leo triplet?",
"What is Regulus?",
"How many light years is 54 Leonis away from Earth?",
"Is the identification of Leo's bright stars a modern discovery?", 
"Why did Hercules fight the Leo lion with his bare hands?",
"Where is Leo located?",
"How is the lion in Leo standing?",
"Why is Leo especially prominent?",
"Is Leo Ring made up of a cloud of hydrogen, helium gas?",
"Is Algieba less than 260 light-years away from Earth?",
"How many constellations were described by Ptolemy?",
"When do the Leonids occur?",
"How many stars of the first or second magnitude does Leo have?",
"How many stars mark the sickle used to represent the Leo constellation?",
"What is Leo's constellation location hemisphere?",
"Is the star SDSS J102915+172927 (Caffau's star) a population II star in the galactic halo seen in Leo?",
"Is the Leo Triplet inside the Leo constellation?",
"What is the traditional name of 40 Leonis?",
"Which star is at the opposite end of the constellation to Alpha Leonis?"]

related_sentences = []

answer_generator = T5AnswerGenerator()

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
    
