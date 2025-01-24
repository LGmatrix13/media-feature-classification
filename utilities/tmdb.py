import requests

class TMDBMovie():
    overview: str
    release_date: str
    vote_average: float
    original_title: str
    genre_ids: list[int]

    def __init__(self, overview: str, release_date: str, vote_average: float, original_title: str, genre_ids: list[int]):
        self.overview = overview
        self.release_date = release_date
        self.vote_average = vote_average
        self.original_title = original_title
        self.genre_ids = genre_ids        

class TMDB():
    @staticmethod
    def read_movies(page: int) -> list[TMDBMovie]:
        r = requests.get(f"https://api.themoviedb.org/3/discover/movie?page={page}", headers={
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjM1MmUyN2FlZjBmNzAzYzE2ZTI5MjA1ZDE0OTRhZiIsIm5iZiI6MTczNzY0OTE1NS4wMjQ5OTk5LCJzdWIiOiI2NzkyNmMwM2U2YzcxYjdkMjZhMDM1YWEiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.m4OBZQAHDk7yiIAJhB080NqOYeblmTfBq5WNR-Ex8so",
            "accept": "application/json"
         })
        data = r.json()
        return [
            TMDBMovie(
                overview=result["overview"],
                release_date=result["release_date"],
                vote_average=result["vote_average"],
                original_title=result["original_title"],
                genre_ids=result["genre_ids"]
            )
            for result in data["results"]
        ]
    
def main():
    movies: list[TMDBMovie] = TMDB.read_movies(page=10)
    print([movie.overview for movie in movies])    

if __name__ == "__main__":
    main()