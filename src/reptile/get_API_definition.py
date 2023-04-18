import requests
from bs4 import BeautifulSoup
import os
import re


if __name__ == '__main__':
    
    # If the file exists, read the contents of the file, otherwise retrieve the lists and store in the file
    filename = 'API_links.txt'
    with open(filename, 'r') as f:
        links_list = f.read().splitlines()
      
    with open("API_def.txt", "w") as f:
        for link in links_list:
            print("Processing " + link + "...")
            response = requests.get(link)
            soup = BeautifulSoup(response.content, "html.parser")
            sig_content = soup.find_all(class_="sig sig-object py")
            for content in sig_content:
                f.write(content.text.strip().replace("\n", "") + " ")
            f.write("\n")

