import csv
import difflib


def get_song(song: str, filename: str) -> str:
    with open(filename, encoding="utf8") as csvfile:
        data = csv.reader(csvfile)
        headers = (next(data))  # Headers

        list_of_songs = [row[0] for row in data]
    close_songs = difflib.get_close_matches(song, list_of_songs)
    similarity = {}
    scores = set()
    for element in close_songs:
        score = difflib.SequenceMatcher(None, element, song).ratio()
        scores.add(score)
        similarity[score] = element
    return similarity[max(scores)]
