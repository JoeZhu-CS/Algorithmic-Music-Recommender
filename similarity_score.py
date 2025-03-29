import numpy as np
from math import sqrt
import get_dict_from_data

def calculate_similarity_score(song1, song2, same_key = False, same_mode = False, tempo_limit = (0,float('inf')), weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]):
    """song1 -> song1 number_data np.array"""
    lower_bound_duration, upper_bound_duration = tempo_limit[0], tempo_limit[1]

    if ((same_key and song1[3] != song2[3]) or  # Key
        (same_mode and song1[5] != song2[5]) or  # Mode
        not (lower_bound_duration <= song2[12] <= upper_bound_duration)):  # Duration
        # Skip song
        pass

    weighted_sum = weights[0] * (song1[0]/100-song2[0]/100)**2  # Normalize popularity to range [0,1]
    for i in [1,2,4, 6, 7, 8, 9, 10]:
        weighted_sum += weights[i] * (song1[i]-song2[i])**2  # weight * (difference squared)

    similarity = sqrt(weighted_sum)  # sqrt((a-b)^2 + (c-d)^2 + ...)
    return similarity





"""
"5000 songs from 6 main categories (EDM, Latin, Pop, R&B, Rap, & Rock)"

track_popularity
    (1) Factor of similarity
    (2) Choose how popular songs should be (most songs on the list are rather well known) 
danceability
    (1) Factor of similarity
energy
    (1) Factor of similarity
key
    (1) Choose whether to only look for songs of the same key
        remember "-1" means that key of the song is unknown 
loudness
    (1) Factor of similarity
mode
    (1) Choose song in same mode (major or minor)
speechiness
    (1) Factor of similarity
acousticness
    (1) Factor of similarity
instrumentalness
    (1) Factor of similarity
liveness
    (1) Factor of similarity
    (2) Only choose live songs (certainty >= 0.8 or smth)
valence
    (1) Factor of similarity
tempo
    (1) Factor of similarity
    (2) Restrict tempo range
duration_ms
    (1) Factor of similarity
    (2) Restrict length of songs
"""

TEST = get_dict_from_data.get_dict_from_data('song_data_clean_no_id_short.csv')
TEST_SONG = TEST["i don't care (with justin bieber) - loud luxury remix"][1]

for item in TEST:
    print(calculate_similarity_score(TEST_SONG))

# Test: "i don't care (with justin bieber) - loud luxury remix"

DICT = list(get_dict_from_data.get_dict_from_data('song_data_clean_no_id_short.csv').values())
print(DICT[2], "\n", DICT[3])

print(calculate_similarity_score(DICT[2][1], DICT[3][1]))
