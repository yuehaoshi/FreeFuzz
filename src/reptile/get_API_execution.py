import requests
from bs4 import BeautifulSoup
import os

# A function to retrive all links for API documents, and store all links into a file named 'links.txt'
def get_lists():
    url = 'https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/api/index_en.html'

    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    menu_items = soup.select('li.toctree-l2 > a.reference.internal')

    links_list = []
    for item in menu_items:
        module = item.text.strip()
        function_items = item.find_next_sibling('ul')
        if function_items is not None:
            function_items = function_items.select('li.toctree-l3 > a.reference.internal')
            for function_item in function_items:
                function_name = function_item.text.strip()
                function_link = 'https://www.paddlepaddle.org.cn' + function_item['href']
                links_list.append(function_link)
    with open('links.txt', 'w') as f:
        f.write('\n'.join(links_list))

if __name__ == '__main__':
    
    # If the file exists, read the contents of the file, otherwise retrieve the lists and store in the file
    filename = 'links.txt'
    if os.path.isfile(filename):
        # Read the contents of the file
        print("File exists, read the contents of the file")
        with open(filename, 'r') as f:
            links_list = f.read().splitlines()
    else:
        print("File does not exist, retrieve the lists and store in the file")
        # Retrieve the lists and store in the file
        links_list = get_lists()
        with open(filename, 'w') as f:
            f.write('\n'.join(links_list))
      
    folder_name = 'code_snippets'
    # create a directory to store the code snippets
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    for link in links_list:
        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        blockquote_tags = soup.find_all('div', {'class': 'highlight-python notranslate'})

        if len(blockquote_tags) == 0:
            print("No tags found in " + link + "")
            continue

        # save each code snippet to a separate .py file
        for i, blockquote in enumerate(blockquote_tags):
            file_name = os.path.join(folder_name, f"script_{link.split('/')[-1]}_{i+1}.py")
            with open(file_name, "w") as f:
                f.write(blockquote.text.strip())