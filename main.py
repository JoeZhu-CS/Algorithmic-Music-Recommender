
from MusicRecommender import MusicRecommender
import os
from custom_weight_selector import customize_weights
# Example usage
if __name__ == "__main__":
    import doctest

    doctest.testmod()

    # Check if the data file exists
    data_file = 'SpotifySongs_no_id.csv'
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        print("Please make sure the file is in the current directory.")
        exit(1)

    # Create a recommender
    recommender = MusicRecommender(data_file)

    # Display welcome message
    print("\n===== Music Recommendation System =====")
    print("This system helps you find music similar to songs you already enjoy.")
    print("You can use basic recommendations or customize your preferences.")

    # Function to find a song with flexible matching
    def find_song(search_term):
        """
        Find the appropriate song based on its title in the csv file
        """

        # Normalize search term (lowercase and strip spaces)
        normalized_search = search_term.lower().strip()

        # Direct match
        if normalized_search in recommender.song_dict:
            return normalized_search

        # Try matching with extra spaces removed
        compressed_search = ''.join(normalized_search.split())
        for song_name in recommender.song_dict.keys():
            compressed_song = ''.join(song_name.split())
            if compressed_search == compressed_song:
                return song_name

        # If still not found, check for partial matches
        potential_matches = []
        for song_name in recommender.song_dict.keys():
            if normalized_search in song_name:
                potential_matches.append(song_name)

        if potential_matches:
            # If we have multiple matches, let the user choose
            if len(potential_matches) > 1:
                print("\nMultiple songs found. Please select one:")
                for i, match in enumerate(potential_matches[:10], 1):  # Show up to 10 matches
                    artist = recommender.song_dict[match][0][0]
                    print(f"{i}. {match.title()} - {artist}")

                while True:
                    try:
                        selection = int(input("\nEnter the number of your choice (or 0 to search again): "))
                        if selection == 0:
                            return None
                        if 1 <= selection <= len(potential_matches[:10]):
                            return potential_matches[selection - 1]
                        print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Please enter a number.")
            else:
                # If we have just one partial match
                return potential_matches[0]

        # No matches found
        return None


    # Main program loop
    while True:
        # Ask user for a song
        print("\nEnter a song name to get recommendations.")
        print("Some popular suggestions: 'Shape of You', 'Despacito', 'Closer', 'Stay'")
        song_input = input("Song title (or 'exit' to quit): ")

        if song_input.lower() == 'exit':
            print("\nThank you for using the Music Recommendation System!")
            break

        song_name = find_song(song_input)

        if not song_name:
            print(f"Sorry, '{song_input}' not found in the dataset.")
            print("Try another song like 'Shape of You' or 'Despacito'")
            continue

        song_title = song_name  # For consistency with the rest of the code
        song_data = recommender.song_dict[song_name]
        artist = song_data[0][0]
        print(f"\nFound: '{song_name.title()}' by {artist}")

        try:
            # Display song features
            print("\n--- Song Feature Analysis ---")
            print("Would you like to see the audio features of this song?")
            if input("Show audio features? (yes/no): ").lower().startswith('y'):
                recommender.visualize_song_features(song_title)
                print("Audio feature visualization displayed.")

            # Ask for preference mode once, upfront
            print("\n--- Recommendation Mode ---")
            print("You can get recommendations in two ways:")
            print("1. Basic mode: Uses default settings to find similar songs")
            print("2. Advanced mode: Lets you set specific preferences (artist, genre, tempo, etc.)")

            while True:
                use_preferences = input("Which mode would you prefer? (basic/advanced): ").strip().lower()

                if use_preferences.startswith('a'):
                    use_preferences = True  # Advanced Mode
                    break

                elif use_preferences.startswith('b'):
                    use_preferences = False  # Basic Mode
                    break

                else:
                    print("Invalid choice. Please enter 'basic' or 'advanced'.")

            # Store preferences if the user wants to use them
            preferences = None
            if use_preferences:
                print("\n--- Setting Your Preferences ---")
                preferences = recommender.get_setting_preferences(song_title)

            # Get recommendations based on chosen mode
            if use_preferences:
                print("\n--- Advanced Preference-Based Recommendations ---")
                similar_songs = recommender.generate_similarity_list(song_title, 10, preferences)
            else:
                print("\n--- Basic Recommendations ---")
                print("Finding similar songs using default parameters...")
                similar_songs = recommender.find_similar_songs(song_title, 5)

            # Display recommendations
            print(f"\nTop {len(similar_songs)} recommendations for '{song_title.title()}':")
            for i, song in enumerate(similar_songs, 1):
                similar_artist = recommender.song_dict[song][0][0]
                similar_genre = recommender.song_dict[song][0][1]
                print(f"{i}. {song.title()} - {similar_artist} (Genre: {similar_genre})")

            # Network visualization - reuse the same preference mode
            print("\n--- Song Network Visualization ---")
            print("You can visualize the network of similar songs.")
            if input("Would you like to see a network visualization? (yes/no): ").lower().startswith('y'):
                print("Creating song similarity network using your previous settings...")
                # Use the same preference mode that was chosen earlier
                recommender.visualize_similar_songs([song_title], max_similar=8, threshold=0.6,
                                                    use_preferences=False if preferences is None else True,
                                                    existing_preferences=preferences)
                print("Network visualization displayed.")

            # Multi-song comparison
            print("\n--- Multi-Song Comparison ---")
            print("You can also compare multiple songs at once.")
            if input("Would you like to add another song for comparison? (yes/no): ").lower().startswith('y'):
                second_song_input = input("Enter another song name: ")
                second_song = find_song(second_song_input)

                if second_song:
                    print(f"Creating network visualization for '{song_title.title()}' and '{second_song.title()}'...")
                    # For multi-song, we should also have the option to reuse preferences
                    multi_use_prefs = input("Use your preferences for this comparison? (yes/no): ").lower().startswith(
                        'y')
                    recommender.visualize_similar_songs([song_title, second_song], max_similar=5,
                                                        use_preferences=multi_use_prefs,
                                                        existing_preferences=preferences if multi_use_prefs else None)
                    print("Multi-song network visualization displayed.")
                else:
                    print(f"Sorry, '{second_song_input}' not found in the dataset.")

            # Filtering recommendations demonstration
            print("\n--- Feature Weights Customization ---")
            print("The recommendation system uses weights for different audio features.")
            if input("Would you like to see the default feature weights? (yes/no): ").lower().startswith('y'):
                # Example feature weights for visualization
                weights = {
                    'genre': 0.15,
                    'artist': 0.05,
                    'popularity': 0.05,
                    'danceability': 0.10,
                    'energy': 0.10,
                    'key': 0.02,
                    'loudness': 0.05,
                    'mode': 0.02,
                    'speechiness': 0.08,
                    'acousticness': 0.08,
                    'instrumentalness': 0.08,
                    'liveness': 0.05,
                    'valence': 0.10,
                    'tempo': 0.05,
                    'duration_ms': 0.02
                }

                print("Showing feature weights visualization...")
                recommender.visualize_feature_weights(weights)
                print("Feature weights visualization displayed.")

                # Demonstrate filtering recommendations
                if input("Would you like to try filtering your recommendations? (yes/no): ").lower().startswith('y'):
                    print("\nFiltering recommendations lets you prioritize specific audio features.")
                    print("For example, we can prioritize danceability and energy for party music.")
                    default_weights = [0.05, 0.12, 0.12, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.05, 0.15, 0.10, 0.05]
                    # Reuse existing preferences or get new ones if needed
                    if preferences is None:
                        print("\nLet's set filtering preferences first:")
                        preferences = recommender.get_setting_preferences(song_title)
                    else:
                        print("Using your previously set preferences with custom weights...")

                    # Use custom weights with preferences
                    custom_songs = recommender.generate_similarity_list(
                        song_title, 10, preferences, default_weights
                    )

                    print(f"\nTop 10 recommendations for '{song_title.title()}':")
                    for i, song in enumerate(custom_songs, 1):
                        similar_artist = recommender.song_dict[song][0][0]
                        print(f"{i}. {song.title()} - {similar_artist}")

            print("\n--- Custom Feature Weights Recommendation ---")
            song_input = input("Enter song title: ").strip().lower()

            try:
                matched_song = find_song(song_input)
                print(f"\nMatched song: {matched_song}")
            except ValueError:
                print("No matching song found. Please check your input and try again.")
                continue

            num_input = input("Enter number of recommendations (default 10): ").strip()
            try:
                num_recs = int(num_input) if num_input else 10
            except ValueError:
                print("Invalid input, using default value 10.")
                num_recs = 10

            if preferences is None:
                print("\nLet's set filtering preferences first:")
                preferences = recommender.get_setting_preferences(song_title)
            else:
                print("Using your previously set preferences with custom weights...")

            custom_weights = customize_weights()
            try:
                rec_songs = recommender.generate_similarity_list(matched_song, n=num_recs, preferences=preferences,
                                                                 weights=custom_weights)
                print("\nRecommended similar songs with custom weights:")
                for idx, song in enumerate(rec_songs, 1):
                    print(f"{idx}. {song}")
            except ValueError as e:
                print(f"Error: {e}")

            if input("Would you like to see the customized feature weights? (yes/no): ").lower().startswith('y'):
                print("Showing feature weights visualization...")
                feature_names = [
                    'popularity', 'danceability', 'energy', 'key', 'loudness',
                    'mode', 'speechiness', 'acousticness', 'instrumentalness',
                    'liveness', 'valence', 'tempo', 'duration'
                ]
                weights_dict = dict(zip(feature_names, custom_weights))
                recommender.visualize_feature_weights(weights_dict)
                print("Feature weights visualization displayed.")

            print("\nWould you like to try another song?")
            if not input("Continue? (yes/no): ").lower().startswith('y'):
                print("\nThank you for using the Music Recommendation System!")
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try something else.")
