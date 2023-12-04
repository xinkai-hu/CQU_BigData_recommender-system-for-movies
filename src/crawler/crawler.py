from bs4 import BeautifulSoup
import requests
import json

base_url = "https://www.rottentomatoes.com"

# To avoid duplicate movie item
movie_list = [""]

# Init all table
with open("people-movie-score.csv", "w") as f:
    f.write("people::movie_name::score\n")
    f.flush()
    f.close()

with open("movie-tags.csv", "w") as f:
    f.write("movie_name::tags\n")
    f.flush()
    f.close()

with open("people-movie-score.csv", "a") as pms_table, open("movie-tags.csv", "a") as mt_table:
    for i in range(1, 14):
        web_source_code = requests.get("https://editorial.rottentomatoes.com/otg-article/new-critics/{}".format(i)).content
        root_dom = BeautifulSoup(web_source_code, 'lxml')
        for j in range(1, 92, 2):
            user_item = root_dom.select(".articleContentBody > p:nth-child(3) > a:nth-child({})".format(j))
            if len(user_item) != 0:
                user_name = str(user_item.pop().attrs['href'].split("/")[-2])
                user_json = dict(json.loads(
                    requests.get("https://www.rottentomatoes.com/napi/critics/{}/movies?pagecount=100".format(user_name))
                    .content))
                user_reviews = user_json.get("reviews")
                print("LOG: Start writing people's review # {}".format(user_name))
                for review in user_reviews:
                    movie_title = review['mediaTitle']
                    movie_score = review['mediaTomatometerScore']
                    movie_sub_url = review['mediaUrl']
                    pms_table.write("{}::{}::{}\n".format(user_name, movie_title, movie_score))
                    print("LOG: Write a review")
                    if movie_list.count(movie_title) == 0:
                        movie_list.append(movie_title)
                        movie_detail_web_url = base_url + movie_sub_url
                        movie_detail_web_source_code = requests.get(movie_detail_web_url).content
                        movie_detail_dom = BeautifulSoup(movie_detail_web_source_code, 'lxml')
                        movie_tags = movie_detail_dom.find("span", {"class": "genre"})
                        if movie_tags is not None:
                            movie_tags = str(movie_tags.contents[0]).strip(" \t\n").replace(' ', '').replace("\n", "")
                            mt_table.write("{}::{}\n".format(movie_title, movie_tags))
                            print("LOG: Movie tags done # {}".format(movie_title))
                mt_table.flush()
                pms_table.flush()
                print("LOG: Finish writing people's review # {}".format(user_name))
