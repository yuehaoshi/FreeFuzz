import requests
from bs4 import BeautifulSoup

if __name__ == '__main__':
    url = "https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/add_en.html"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    blockquote_tags = soup.find_all('div', {'class': 'highlight-python notranslate'})

    if len(blockquote_tags) == 0:
        print("Notags found.")
        exit(1)
    else:
        print(f"Found {len(blockquote_tags)} blockquote tags.")
    for blockquote in blockquote_tags:
        print(blockquote.text)
