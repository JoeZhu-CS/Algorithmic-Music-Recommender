import numpy as np
import csv


def get_dict_from_data(filename: str) -> dict[str, list]:
    """
    Returns a python dictionary from the data contained in the CSV file <filename>.

    Structure (of return value):
        - One large python dictionary array containing one python list per song.
        - The key is the song name in lowercase
        - Each corresponding value is a python list containing two numpy arrays.
        The first array contains text_data (artist name, genre, etc.)
        while the second array contains number_data (popularity, loudness, key, etc.).

    ORDER:
        {track_name: [np.array([track_artist, playlist_genre]), np.array([track_popularity,
        danceability, energy, key, loudness, mode, speechiness, acousticness,
        instrumentalness, liveness, valence,tempo, duration_ms])]}

    Preconditions:
        Input CSV data must look like the cleaned no_id dataset (see GitHub for reference):
        [track_name, track_artist, track_popularity, playlist_genre,
        danceability, energy, key, loudness, mode, speechiness, acousticness,
        instrumentalness, liveness, valence,tempo, duration_ms]
    """
    text_indices = [1, 3]
    number_indices = [2] + list(range(4, 16))

    final_dict = {}
    with open(filename, encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            key = row[0].lower()
            text_data = np.array([row[i] for i in text_indices])
            number_data = np.array([float(row[i]) for i in number_indices])
            final_dict[key] = [text_data, number_data]
    return final_dict
