def format_llava_prompt(prompt_original: str) -> str:
    """Format LLaVA-1.5 prompts consistently for training and evaluation."""
    return f"USER: <image>\n{prompt_original}\nASSISTANT:"
