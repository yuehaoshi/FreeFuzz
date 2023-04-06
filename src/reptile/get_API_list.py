# soup = BeautifulSoup(response.content, 'html.parser')

# # Find the sidebar section using CSS selectors or other methods
# sidebar = soup.find('div', class_='paddle-doc-page-left-menu-wrap')

# # Extract the indexing contents from the sidebar section
# indexing_contents = []
# for item in sidebar.find_all('a'):
#     indexing_contents.append(item.text)

# # Print the indexing contents
# print(indexing_contents)
import requests
from bs4 import BeautifulSoup

url = 'https://www.paddlepaddle.org.cn/documentation/docs/en/2.2/api/index_en.html'

response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
# print("soup: ", soup)
menu_items = soup.select('li.toctree-l2 > a.reference.internal')

menu_list = []
for item in menu_items:
    module = item.text.strip()
    function_items = item.find_next_sibling('ul')
    if function_items is not None:
        function_items = function_items.select('li.toctree-l3 > a.reference.internal')
        for function_item in function_items:
            function_name = function_item.text.strip()
            menu_list.append(module + '.' + function_name)

# print(menu_list)
with open('API_lists.txt', 'w') as f:
    f.write('\n'.join(menu_list))
