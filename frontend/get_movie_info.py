import requests
from bs4 import BeautifulSoup
import asyncio
from copy import deepcopy

class Movie:
    def __init__(self, name):
        self.name = name
        self.movie_name = deepcopy(name)
        self.res = []
        self.final_res = []

    async def get_movie_info(self, name):
        search_url = 'https://www.imdb.com/find?ref_=nv_sr_fn&q=' + name + '&s=all'
        search_html = requests.get(search_url).text
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        search_html = requests.get(search_url, headers=headers).text

        search_page = BeautifulSoup(search_html, 'lxml')
        match = search_page.find('li', class_="find-result-item")
        # print(match)
        if match:
            await self.get_detail(match, name)
            await asyncio.sleep(1)  # Add a 1-second delay to avoid excessive requests


    async def get_detail(self, match, name):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        result = dict()

        try:
            poster_url = match.find('img')['src']
        except:
            poster_url = None

        summary = None
        imdb_rating = None
        time = None
        stars = None
        director = None

        try:
            movie_url = match.find('a', class_='ipc-metadata-list-summary-item__t')['href']
            url = 'https://www.imdb.com' + movie_url
            res = requests.get(url, headers=headers).text
            page = BeautifulSoup(res, 'lxml')
            # Extract IMDb rating
            rating_element = page.find('div', class_='sc-acdbf0f3-0')
            # print("rating element =================", str(rating_element) + ' \n')
            imdb_rating = rating_element.find('span', class_='sc-bde20123-1').text.strip()
            stars = rating_element.find('div', class_='sc-bde20123-3').text.strip()
            
            # extract summary
            documentary_element = page.find('div', class_='sc-e226b0e3-10')
            summ_element = documentary_element.find('p', class_='sc-7193fc79-3')
            summary = summ_element.find('span', class_='sc-7193fc79-0').text.strip()

            # Extract movie runtime
            runtime_element = page.find('li', class_='ipc-inline-list__item', string=lambda text: 'h' in text and 'm' in text)
            time = runtime_element.text.strip() if runtime_element else None


            # Extract director
            director_element = page.find('a', href=lambda href: href and '/name/nm' in href)
            director = director_element.text.strip() if director_element else None

        except:
            summary = None
            imdb_rating = None
            time = None
            stars = None
            director = None

        result['poster'] = poster_url
        result['summary'] = summary
        result['ratings'] = imdb_rating
        result['time'] = time
        result['director'] = director
        result['stars'] = stars
        result['movie_name'] = name
        self.res.append(result)

        # print("results-----------------------", self.res)

    async def process_movies(self):
        tasks = [self.get_movie_info(name) for name in self.movie_name]
        await asyncio.gather(*tasks)
        self.re_order()

    def re_order(self):
        for i in range(len(self.movie_name)):
            for j in range(len(self.res)):
                if self.movie_name[i] == self.res[j]['movie_name']:
                    self.final_res.append(self.res[j])

def get_movie_info(movie_list):
    movie_instance = Movie(movie_list)
    asyncio.run(movie_instance.process_movies())
    return movie_instance.final_res

# movie_list = ['maze runner', 'the matrix', 'destroyer', 'The Shawshank Redemption', 'The Godfather']

# import time
# head = time.time()
# movies_info = get_movie_info(movie_list)
# end = time.time()
# print(movies_info)
# print(f"Execution time: {end - head:.2f} seconds")
