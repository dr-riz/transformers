import torch
from transformers import BertModel

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertModel.from_pretrained("bert-base-uncased")

#
sequence = "Using a Transformer network is simple" 
print("tokenize sequence: ", sequence)
tokens = tokenizer.tokenize(sequence)
print("tokens: ", tokens)


# example_token_id = tokenizer.convert_tokens_to_ids(["example"])[0]
example_token_id = tokenizer.convert_tokens_to_ids(tokens)
print("token ids for the tokens: ",  example_token_id)


print("get the embedding vector for the token ids: ", example_token_id)
example_embedding = model.embeddings.word_embeddings(torch.tensor([example_token_id]))

print(example_embedding)
print(example_embedding.shape)
# torch.Size([1, 768])