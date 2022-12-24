from nltk.translate.bleu_score import sentence_bleu




fr_ref = open('FlanT5_Domain_knowledge_references','r')
fr_pred = open('FlanT5_Domain_knowledge_predicted','r')

references = fr_ref.readlines()
predicted_results = fr_pred.readlines()



scores = []
BLEU2 = []
for i in range(len(references)):
	score = sentence_bleu(references[i].strip().split(), predicted_results[i].strip().split(), weights=(1.0, 0.0, 0.0, 0.0))
	bleu2 = sentence_bleu(references[i].strip().split(), predicted_results[i].strip().split(), weights=(0.5, 0.5, 0.0, 0.0))
	scores.append(score)
	BLEU2.append(bleu2)

print("BLEU-1 score:"+str(sum(scores)/len(scores)))
print("BLEU-2 score:"+str(sum(BLEU2)/len(BLEU2)))