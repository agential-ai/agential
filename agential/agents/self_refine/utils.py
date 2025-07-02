from rich.console import Console
from rich.panel import Panel
from rich.markup import escape


def log_llm_io(
    response, context: str = "", verbose: bool = False, truncate_length: int = -1
):
    """
    Log LLM input/output with rich formatting.

    Args:
        response: LLM response object
        context: Context string for the log
        verbose: Whether to log
        truncate_length: Maximum length before truncating
    """
    if not verbose:
        return
    console = Console()
    input_text = escape(str(response.input_text))
    output_text = escape(response.output_text)
    if truncate_length != -1 and truncate_length > 0:
        if len(input_text) > truncate_length:
            input_text = input_text[:truncate_length] + "..."
        if len(output_text) > truncate_length:
            output_text = output_text[:truncate_length] + "..."
    context_escaped = escape(context)
    content = f"[bold blue]LLM {context_escaped}[/bold blue]\n\n[bold green]INPUT:[/bold green]\n{input_text}\n\n[bold yellow]OUTPUT:[/bold yellow]\n{output_text}"
    console.print(Panel(content, title="🤖 LLM Call", border_style="blue"))
