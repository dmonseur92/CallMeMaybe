import json
from llm_sdk import Small_LLM_Model

def read_prompt() -> list[str]:
    file = "data/input/function_calling_tests.json"
    prompts = []
    with open(file) as f:
        data = json.load(f)
        for d in data:
           prompts.append(d["prompt"])
    return prompts

def read_def() -> list[dict]:
    file = "data/input/functions_definition.json"
    with open(file) as f:
        data = json.load(f)
    return data

def lets_go():
    model = Small_LLM_Model()
    prompts = read_prompt()
    definitions = read_def()
    definitions_text = json.dumps(definitions)
    # print(definitions_text)
    for p in prompts:
        text = definitions_text + "\nUser request: Give me the function name you are using to answer " + p
        tokens = model.encode(text)

        input_ids = tokens[0].tolist()
        i = 0
        while i < 50 :
            logits = model.get_logits_from_input_ids(input_ids)
            input_ids.append(logits.index(max(logits)))
            i += 1

        # print(input_ids)
        print(model.decode(input_ids))
        print()

if __name__ == "__main__":
    lets_go()
