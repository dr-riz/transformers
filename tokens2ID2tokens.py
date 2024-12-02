from transformers import AutoTokenizer 

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name) 

sequence = "Using a Transformer network is simple" 
print(sequence)
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids) 
decoded_string = tokenizer.decode(ids) 
print(decoded_string) 

res=tokenizer(sequence)
print(res)