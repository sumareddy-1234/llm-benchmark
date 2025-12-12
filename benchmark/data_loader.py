def load_prompts(file_path):
    """
    Load prompts from a text file. Each line is considered a separate prompt.
    """
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts
