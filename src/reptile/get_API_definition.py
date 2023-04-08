import requests
from bs4 import BeautifulSoup
import os
import re

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
      
    with open("API_def.txt", "w") as f:
        for link in links_list:
            print("Processing " + link + "...")
            response = requests.get(link)
            soup = BeautifulSoup(response.content, "html.parser")
            sig_content = soup.find_all(class_="sig sig-object py")
            for content in sig_content:
                f.write(content.text.strip().replace("\n", "") + " ")
            f.write("\n")