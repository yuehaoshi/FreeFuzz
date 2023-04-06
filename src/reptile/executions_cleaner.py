# import re

# with open('output.txt', 'r') as f:
#     # Read the contents of the file
#     code = f.read()

#     # Remove comments
#     code = re.sub(r'#.*', '', code)

#     # Extract import statements
#     imports = set(re.findall(r'^import .*$', code, flags=re.MULTILINE))
#     from_imports = set(re.findall(r'^from .* import .*$', code, flags=re.MULTILINE))

#     # Remove duplicate imports
#     code = re.sub(r'^import .*$', '', code, flags=re.MULTILINE)
#     code = re.sub(r'^from .* import .*$', '', code, flags=re.MULTILINE)
#     code = '\n'.join(list(imports.union(from_imports))) + '\n' + code

#     # Replace more than 2 continuous empty lines with two empty lines
#     code = re.sub(r'\n{3,}', '\n\n', code)

#     # Write the modified code back to the file
#     with open('modified_execution.txt', 'w') as f2:
#         f2.write(code)
import re

with open('output.txt', 'r') as f:
    # Read the contents of the file
    code = f.read()

    # Remove comments
    code = re.sub(r'#.*', '', code)

    # Extract import statements
    imports = set(re.findall(r'^import .*$', code, flags=re.MULTILINE))
    from_imports = set(re.findall(r'^from .* import .*$', code, flags=re.MULTILINE))

    # Remove duplicate imports
    code = re.sub(r'^import .*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^from .* import .*$', '', code, flags=re.MULTILINE)
    
    # Combine import statements and move "import" before "from .. import .."
    import_statements = sorted(list(imports.union(from_imports)))
    import_lines = [stmt for stmt in import_statements if stmt.startswith("import")]
    from_import_lines = [stmt for stmt in import_statements if stmt.startswith("from")]
    code = '\n'.join(import_lines + from_import_lines) + '\n' + code
    
    # Replace more than 2 continuous empty lines with 2 empty lines
    code = re.sub(r'\n{3,}', '\n\n', code)

    # Write the modified code back to the file
    with open('modified_execution.txt', 'w') as f2:
        f2.write(code)