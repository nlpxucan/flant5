from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import json



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")

model.parallelize({
		    0: [0,1, 2],
		    1: [3, 4, 5],
		    2: [6, 7, 8],
		    3: [9, 10, 11],
		    4: [12, 13, 14],
		    5: [15, 16, 17],
		    6: [18, 19, 20],
		    7: [21, 22, 23]
		})


fr = open('Techspec_Wiki_Bing_Sum_test.jsonl','r')
fw = open('FlanT5_Domain_knowledge_predicted','w')
fw_ref = open('FlanT5_Domain_knowledge_references','w')
lines = fr.readlines()



count = 0

for line in lines:
	print(count)
	count += 1

	line_obj = json.loads(line.strip())
	
	input_ids = tokenizer(line_obj['input'], return_tensors="pt").input_ids.to(device)
	outputs = model.generate(input_ids,max_length=2300,early_stopping=True)
	predicted_result = tokenizer.decode(outputs[0]).replace("<pad>", "").replace("</s>", "")	

	fw.write(predicted_result+'\n')
	fw_ref.write(line_obj['output']+'\n')

fw.close()
fw_ref.close()