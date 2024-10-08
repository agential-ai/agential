import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Example of default value being None")

# Add an argument with a default value of None
parser.add_argument('--my_arg', default=None, help='This argument defaults to None')

# Parse the arguments
args = parser.parse_args()

# Access the argument value
print(f'The value of my_arg is: {args.my_arg}')
