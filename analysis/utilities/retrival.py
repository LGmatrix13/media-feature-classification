import pandas as pd
import tmdb

NUM_MOVIES = 200000
pages = NUM_MOVIES // 100
movies_list = []  # Use a list to store dictionaries

print("Running retrival to TMDB...")

for i in range(1, pages + 1):
    print(f'Runnning page {i}...')
    movies = tmdb.Client.read_movies(page=i)
    for movie in movies:
        movies_list.append(movie.to_dict())

result = pd.DataFrame(movies_list)
result.to_csv("../../data/digital/tmdb.csv")
