def delete_char(sentence):
    first_comma_index = sentence.find(',')
    last_comma_index = sentence.rfind(',')

    if first_comma_index != -1 and last_comma_index != -1:
        prompt = sentence[first_comma_index + 2: last_comma_index]

        return prompt
    else:
        raise Exception("Wrong Data Format!")


def load_prompt(path, prompt_version):
    if prompt_version == 'pick':
        prompts = []
        seeds = []
        with open(path, 'r', encoding='utf-8') as file:
            contents = file.readlines()

            for row in contents:
                prompt = delete_char(row)
                seed = row.split(' ')[-1]

                prompts.append(prompt)
                seeds.append(seed)

        return prompts, seeds
    

def load_pick_prompt(path):
    prompts = []
    seeds = []
    with open(path, 'r', encoding='utf-8') as file:
        content = file.readlines()
        
        for row in content:
            prompts.append(eval(row)['caption'])
            seeds.append(eval(row)['seed'])
        
    return prompts, seeds
