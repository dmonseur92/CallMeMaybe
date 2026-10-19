import json
from typing import Any
from llm_sdk import Small_LLM_Model

def read_prompt() -> list[str]:
    file = "data/input/function_calling_tests.json"
    prompts = []
    with open(file) as f:
        data = json.load(f)
        for d in data:
           prompts.append(d["prompt"])
    return prompts

def read_def() -> dict[Any]:
    file = "data/input/functions_definition.json"
    definitions = []
    with open(file) as f:
        data = json.load(f)
        for d in data:
            #print(d["name"])
            definitions.append(d["description"])
            #print(d["parameters"])
    return definitions

def lets_go():
    model = Small_LLM_Model()
    prompts = read_prompt()
    for p in prompts:
        tokens = model.encode(p)
        print(tokens)
        print(model.decode(tokens))

        input_ids = tokens[0].tolist()
        i = 0
        while i < 10 :
            logits = model.get_logits_from_input_ids(input_ids)
            input_ids.append(logits.index(max(logits)))
            i += 1

        print(input_ids)
        print()
        print(model.decode(input_ids))

if __name__ == "__main__":
    lets_go()
