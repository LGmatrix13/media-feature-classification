from time import sleep
import pandas as pd
from utilities.tmdb import TMDB

def main(verbose: bool = True):
    NUM_MOVIES = 200000
    pages = NUM_MOVIES // 100
    movies_list = [] 
    if verbose: print("Running retrival to TMDB...")

    for i in range(1, pages + 1):
        if verbose: print(f'Runnning page {i}...')
        movies = TMDB.read_movies(page=i)
        for movie in movies:
            movies_list.append(movie.to_dict())
        sleep(0.5)

    result = pd.DataFrame(movies_list)
    result.to_csv('tmdb.csv')

if __name__ == "__main__":
    main()
