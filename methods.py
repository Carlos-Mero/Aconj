from openai import OpenAI
from datasets import load_dataset

# format_setting = """{"type"}"""
format_setting = {"type": "json_object"}

def run_naive(config):
    ds = load_dataset(config['dataset'], config['data_split']).shuffle(seed=config['seed'])
    train_split = ds['train']

    if config['dataset'] == 'lighteval/MATH' and config['catagory'] != 'All':
        train_split = train_split.filter(lambda x: x['type'] == config['catagory'])

    messages = [
        {"role": "system", "content": config['system_prompt']},
    ]

    for i in range(config['n_samples']):
        qhead = config['ans_prop']
        if config['dataset'] == "gsm8k":
            prompt = qhead + train_split['question'][i]
        elif config['dataset'] == "lighteval/MATH":
            prompt = qhead + train_split['problem'][i]

        print(f"Using prompt: {prompt}")

        messages.append({"role": "user", "content": prompt})

        client = OpenAI().chat.completions
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
            response_format=format_setting
        )

        cnt = completion.choices[0].message.content
        print(cnt)
        messages.append({"role": "assistant", "content": cnt})

        if config['dataset'] == "lighteval/MATH":
            print(f"The true answer is:\n{train_split['solution'][i]}")
            print(f"The question is of difficulty: {train_split['level'][i]}\nand with type: {train_split['type'][i]}")

    messages.append({"role": "user", "content": config['conj_prop']})
    print(f"Using prompt: {config['conj_prop']}")
    completion = client.create(
        model=config['model'],
        messages=messages,
        temperature=config['temperature'],
        seed=config['seed'],
        response_format=format_setting
    )
    cnt = completion.choices[0].message.content
    print(cnt)

def run_review(config):
    ds = load_dataset(config['dataset'], config['data_split']).shuffle(seed=config['seed'])
    train_split = ds['train']
    if config['dataset'] == 'lighteval/MATH' and config['catagory'] != 'All':
        train_split = train_split.filter(lambda x: x['type'] == config['catagory'])

    messages = [
        {"role": "system", "content": config['system_prompt']},
    ]

    for i in range(config['n_samples_perreview']):
        qhead = config['ans_prop']
        if config['dataset'] == "gsm8k":
            prompt = qhead + train_split['question'][i]
        elif config['dataset'] == "lighteval/MATH":
            prompt = qhead + train_split['problem'][i]

        print(f"Using prompt: {prompt}")

        messages.append({"role": "user", "content": prompt})

        client = OpenAI().chat.completions
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
            response_format=format_setting
        )

        cnt = completion.choices[0].message.content
        print(cnt)
        messages.append({"role": "assistant", "content": cnt})

        if config['dataset'] == "lighteval/MATH":
            print(f"The true answer is:\n{train_split['solution'][i]}")
            print(f"The question is of difficulty: {train_split['level'][i]}\nand with type: {train_split['type'][i]}")

    messages.append({"role": "user", "content": config['conj_prop']})
    print(f"Using prompt: {config['conj_prop']}")
    completion = client.create(
        model=config['model'],
        messages=messages,
        temperature=config['temperature'],
        seed=config['seed'],
        response_format=format_setting
    )
    cnt = completion.choices[0].message.content
    print(cnt)

    messages.append({"role": "assistant", "content": cnt})

    for i in range(config['n_reviews']):
        messages.append({"role": "user", "content": config['review_prop']})
        messages[0] = {"role": "system", "content": config['reviewer_sys_prop']}
        print(f"Using prompt: {config['review_prop']}")
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
            response_format=format_setting
        )
        cnt = completion.choices[0].message.content
        print(cnt)
        messages.append({"role": "assistant", "content": cnt})

        messages.append({"role": "user", "content": config['conj_prop']})
        print(f"Using prompt: {config['conj_prop']}")
        completion = client.create(
            model=config['model'],
            messages=messages,
            temperature=config['temperature'],
            seed=config['seed'],
            response_format=format_setting
        )
        cnt = completion.choices[0].message.content
        messages.append({"role": "assistant", "content": cnt})
        print(cnt)
