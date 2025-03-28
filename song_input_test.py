import csv
import difflib


def get_song(song: str, artist: str, filename: str) -> str:
    """
    Return the song in <filename> that is most similar to <song> by name and artist, which are inputted by the user.

    If no similar songs are found, return 'No song found'.

    Preconditions:
        - filename is in the format of the Spotify song dataset (see Github for reference)
    """
    with open(filename, encoding="utf8") as csvfile:
        data = csv.reader(csvfile)
        headers = (next(data))  # Headers
        rows = list(data)  # Converting csv data to a list of lists, we can iterate through data more than once
        list_of_songs = [row[0] for row in rows]
        list_of_songs_in_lowercase = [row[0].lower() for row in rows]
        list_of_artists = [row[1].lower() for row in rows]

    close_songs = difflib.get_close_matches(song.lower(), list_of_songs_in_lowercase)  # Song titles similar to user's input
    close_artists = difflib.get_close_matches(artist.lower(), list_of_artists)  # Song artists similar to user's input

    if not close_songs or not close_artists:  # There are no close songs or artists to the user's input
        return 'No song found'

    score_to_song = {}
    score_to_artist = {}
    for i in range(len(list_of_artists)):
        if list_of_songs_in_lowercase[i] in close_songs and list_of_artists[i] in close_artists:
            # Both the song and artist are similar to the user's inputs

            # Return how similar a song is to the user's input, textwise
            score = difflib.SequenceMatcher(None, list_of_songs_in_lowercase[i], song.lower()).ratio()
            # Mapping how similar a song is to the user's input to the song itself
            score_to_song[score] = list_of_songs[i]
            score_to_artist[score] = list_of_artists[i]

    # None of the songs/artists were similar
    if not score_to_song or not score_to_artist:
        return 'No song found'

    # Return the most similar song
    return score_to_song[max(score_to_song.keys())] # + " by " + score_to_artist[max(score_to_artist.keys())]
