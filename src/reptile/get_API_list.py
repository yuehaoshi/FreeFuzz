import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Define the base URL of the documentation website
base_url = 'https://www.paddlepaddle.org.cn/documentation/docs/en/api/'

# Join the base URL with the relative link of the index page
url = urljoin(base_url, 'index_en.html')
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find all the menu items in the index page
menu_items = soup.select('li.toctree-l2 > a.reference.internal')

# Initialize empty lists to store the API names and links
menu_list = []
link_list = []

# Loop over each menu item
for item in menu_items:
    # Get the module name from the menu item text
    module = item.text.strip()

    # Find the function items in the next sibling ul element
    function_items = item.find_next_sibling('ul')

    # If there are function items, loop over each of them
    if function_items is not None:
        function_items = function_items.select('li.toctree-l3 > a.reference.internal')
        for function_item in function_items:
            # Get the function name from the function item text
            function_name = function_item.text.strip()

            # Join the base URL with the relative link of the API page
            link = urljoin(base_url, function_item['href'])

            # If the link starts with the base URL, add the module name and function name to the menu list,
            # and add the link to the link list
            if link.startswith(base_url):
                menu_list.append(module + '.' + function_name)
                link_list.append(link)

# Write the API names to a file
with open('API_lists.txt', 'w') as f:
    f.write('\n'.join(menu_list))

# Write the API links to a file
with open('API_links.txt', 'w') as f:
    f.write('\n'.join(link_list))
