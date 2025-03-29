import numpy as np


def similarity_main(song_name:str, main_dictionary:dict, setting_preferences = None | tuple[bool, bool, bool, bool, float, float, float, float]):
    """
    Takes in a song as a key-value pair (tuple) from the main song dictionary.
    """
    main_song = main_dictionary[song_name]

    if setting_preferences is None:
        setting_preferences = get_setting_preferences((song_name,main_dictionary[song_name]))

    # Main Loop
    """
    top_list = [] (of length n, if user chooses n songs)
    for song in main_dict:
        calculate score
        check if it's better than the songs we have 
        add it to the list of songs (so that the list remains sorted)
    return the list of songs
    """

def get_setting_preferences(song:tuple[str,list]) -> tuple[bool, bool, bool, bool, float, float, float, float]:
    """
    Returns the user's preferred settings for the search. Takes in a song as a key-value pair (tuple) from the main song dictionary.
    RETURN ORDER:
    - same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration
    """

    # === Same Artist ===
    answer = input("Restrict songs to those from the same artist? \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same artist? \nInput Yes or No \n> ").lower()
    same_artist = {"yes": True, "no": False}[answer]

    # === Same Genre ===
    answer = input("Restrict songs to those from the same genre (estimated genre from playlist information)? \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same genre? \nInput Yes or No \n> ").lower()
    same_genre = {"yes": True, "no": False}[answer]

    # === Same Key ===
    answer = input("Restrict songs to those of the same key? \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same key? \nInput Yes or No \n> ").lower()
    same_key = {"yes": True, "no": False}[answer]

    # === Same Mode ===
    answer = input("Restrict songs to those of the same mode (Major/Minor)? \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same mode? \nInput Yes or No \n> ").lower()
    same_mode = {"yes": True, "no": False}[answer]

    # === Restrict Tempo ===
    answer = input(f"Restrict songs based on tempo?  For reference, your chosen song, \"{song[0]}\", has a tempo of {song[1][11]} bpm. \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same mode? \nInput Yes or No \n> ").lower()
    if {"yes": True, "no": False}[answer]:
        # === Lower Bound ===
        answer = input("Input a lower bound for tempo (bpm). (integers only) \n> ")
        while not answer.isdigit():
            print("Invalid Answer. Please try again.")
            answer = input("Input a lower bound for tempo (bpm). \n> ")
        lower_bound_tempo = float(answer)
        # === Upper Bound ===
        answer = input("Input an upper bound for tempo (bpm). (integers only) \n> ")
        while not answer.isdigit():
            print("Invalid Answer. Please try again.")
            answer = input("Input an upper bound for tempo (bpm). \n> ")
        upper_bound_tempo = float(answer)
    else:
        lower_bound_tempo = 0
        upper_bound_tempo = float('inf')  # No upper limit

    # === Restrict Duration ===
    answer = input(f"Restrict songs based on song duration?  For reference, your chosen song, \"{song[0]}\", has a duration of {int(song[1][12]/1000)} seconds, or around {round(song[1][12]/1000/60, 3)} minutes. \nInput Yes or No \n> ").lower()
    while answer not in ["yes", "no"]:
        print("Invalid Answer. Please try again.")
        answer = input("Same mode? \nInput Yes or No \n> ").lower()
    if {"yes": True, "no": False}[answer]:
        # === Lower Bound ===
        answer = input("Input a lower bound for duration (seconds). (integers only) \n> ")
        while not answer.isdigit():
            print("Invalid Answer. Please try again.")
            answer = input("Input a lower bound for tempo (seconds). \n> ")
        lower_bound_duration = float(answer)*1000  # Convert to milliseconds
        # === Upper Bound ===
        answer = input("Input an upper bound for tempo (seconds). (integers only) \n> ")
        while not answer.isdigit():
            print("Invalid Answer. Please try again.")
            answer = input("Input an upper bound for tempo (seconds). \n> ")
        upper_bound_duration = float(answer)*1000  # Convert to milliseconds
    else:
        lower_bound_duration = 0
        upper_bound_duration = float('inf')  # No upper limit

    return same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration