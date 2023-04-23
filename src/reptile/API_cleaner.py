import re
with open('API_def.txt', 'r') as f:
    lines = f.readlines()

modified_lines = []
modified_lines_without_parentheses = []
for text in lines:
    # Remove spaces before or after a hyphen
    text = re.sub(r"\s*-\s*", "-", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    if not text.strip().startswith("paddle"):
        continue

    # Remove everything after "[source]"
    text = re.sub(r"\[source\].*$", "", text)

    # Skip lines that do not end with ")"
    if not text.strip().endswith(")"):
        continue

    text_without_parentheses = re.sub(r"\(.*\)", "", text)

    # Skip empty lines
    if not text.strip():
        continue

    modified_lines.append(text)
    modified_lines_without_parentheses.append(text_without_parentheses)

with open('API_def_mod.txt', 'w') as f:
    f.write('\n'.join(modified_lines))

with open('API_lists.txt', 'w') as f:
    f.write('\n'.join(modified_lines_without_parentheses))