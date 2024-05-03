"""Utility function for running python code and returning the executed code or error."""


def execute(input_code: str) -> str:
    """Takes in Python code and executes, returning the namespace.

    Args:
        input_code (str): String representation of lines of executable python.

    Returns:
        str: A string containing the variables and values from the executed code.

    Example:
        code = "a = 1*100"
        execute(code)
        # "a = 100"
    """
    local_vars = {}
    print(f"INPUT CODE AT BEGINNING: {input_code}")
    input_code = input_code.split(";")
    input_code = [line.strip() for line in input_code]
    input_code = "\n".join(input_code)
    print(f"INPUT CODE ABOUT TO BE PARSED: {input_code}")
    try:
        exec(input_code, globals(), local_vars)
    except Exception as e:
        return str(e)

    # Convert the local namespace to a dictionary
    variables_dict = {
        var: local_vars[var] for var in local_vars if not var.startswith("__")
    }

    # Convert the dictionary to a string representation
    result = ", ".join([f"{key} = {value}" for key, value in variables_dict.items()])
    print(f"LOCAL_VARS = {local_vars}")

    return result


if __name__ == "__main__":
    # Example usage:
    code = """total_pages = 15 #Sheila has a 15-page research paper;finished_pages = total_pages * (1/3) #She already finished 1/3 of the paper;pages_left = total_pages - finished_pages #The pages left to write is the total minus the finished pages"""
    print(execute(code))
