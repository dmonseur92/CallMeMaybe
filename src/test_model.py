from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()
tokens = model.encode("What is the sum of 2 and 3?")
print(tokens)

print(model.decode(tokens))

input_ids = tokens[0].tolist()
logits = model.get_logits_from_input_ids(input_ids)

print(len(logits))
print(max(logits))
print(logits.index(max(logits)))
print(model.decode([3555]))

input_ids.append(logits.index(max(logits)))
logits = model.get_logits_from_input_ids(input_ids)
print(model.decode([logits.index(max(logits))]))

# uv run --project llm_sdk python src/main.py
