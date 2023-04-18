import requests
from bs4 import BeautifulSoup
import os

if __name__ == '__main__':

    # If the file exists, read the contents of the file, otherwise retrieve the lists and store in the file
    filename = 'API_links.txt'
    with open(filename, 'r') as f:
        links_list = f.read().splitlines()

    folder_name = 'code_snippets'

    # create a directory to store the code snippets if it does not exist
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # loop over each link in the list of links
    for link in links_list:
        # print the website's name in the terminal
        print(f"Processing website: {link}")

        response = requests.get(link)
        soup = BeautifulSoup(response.content, 'html.parser')
        blockquote_tags = soup.find_all('div', {'class': 'highlight-python notranslate'})

        if len(blockquote_tags) == 0:
            print("No tags found in " + link + "")
            continue

        # save each code snippet to a separate .py file
        for i, blockquote in enumerate(blockquote_tags):
            file_name = os.path.join(folder_name, f"script_{link.split('/')[-1]}_{i + 1}.py")
            with open(file_name, "w") as f:
                f.write(blockquote.text.strip())
