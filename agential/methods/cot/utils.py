def log_llm_io(response, label, verbose, truncate_length=-1):
    if not verbose:
        return
    print(f"\n[{label}]\n{'-' * 40}")
    print(
        f"Prompt:\n{response.prompt[:truncate_length] if truncate_length > 0 else response.prompt}"
    )
    print(
        f"Output:\n{response.output_text[:truncate_length] if truncate_length > 0 else response.output_text}"
    )
    print(f"Tokens: {response.total_tokens}, Cost: ${response.total_cost:.6f}")
