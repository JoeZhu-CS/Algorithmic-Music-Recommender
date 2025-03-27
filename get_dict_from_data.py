import numpy as np
import csv

def get_dict_from_data(filename:str) -> dict[str, list]:
    """
    Returns a python dictionary from the data contained in the CSV file <filename>.

    Structure (of return value):
        - One large python dictionary array containing one python list per song.
        - The key is the song name in lowercase
        - Each corresponding value is a python list containing two numpy arrays. The first array contains text_data (artist name, genre, etc) while the second array contains number_data (popularity, loudness, key, etc).

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
    final_dict = {}
    with open(filename, encoding="utf8") as csvfile:
        
        data = csv.reader(csvfile)
        headers = (next(data))  # Headers

        for row in data:
            # Text data
            text_data = np.array([row[1], row[3]])
            # Number data
            number_data = np.array([float(row[2]), float(row[4]), float(row[5]), float(row[6]), float(row[7]), float(row[8]), float(row[9]), float(row[10]),float(row[11]), float(row[12]), float(row[13]), float(row[14]), float(row[15])])
            parsed_row = [text_data, number_data]  # Add text and number data to a list (1 per song)
            final_dict[row[0].lower()] = parsed_row  # Key = lowercase song name, value = parsed_row
    
    return final_dict
