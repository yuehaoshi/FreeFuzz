import re
with open('API_def_ori.txt', 'r') as f:
    lines = f.readlines()

modified_lines = []
for text in lines:
    # Remove spaces before or after a hyphen
    text = re.sub(r"\s*-\s*", "-", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    # Remove "class " at the beginning of the line, if present
    text = re.sub(r"^class\s+", "", text)

    # Remove everything after "[source]"
    text = re.sub(r"\[source\].*$", "", text)

    # Skip lines that do not end with ")"
    if not text.strip().endswith(")"):
        continue
        
    # Skip empty lines
    if not text.strip():
        continue

    modified_lines.append(text)

with open('API_def_mod.txt', 'w') as f:
    f.write('\n'.join(modified_lines))