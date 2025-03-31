import os
import sys

from visualization import MusicRecommender
from song_input_test import get_song
from custom_weight_selector import customize_weights


def main():

    """
    The entry point for the Algorithmic Music Recommender project.
    This script loads the necessary CSV dataset, integrates all the various modules
    and provides a menu-driven command-line interface.
    """

    print("==========================================")
    print("  Welcome to Algorithmic Music Recommender")
    print("==========================================\n")

    data_file = "SpotifySongs_no_id.csv"

    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        sys.exit(1)

    recommender = MusicRecommender(data_file)

    # 主菜单循环
    while True:
        print("\nPlease choose an option:")
        print("1. Get song recommendations with default settings")
        print("2. Get song recommendations with custom settings")
        print("3. View song feature analysis")
        print("4. Visualize song similarity network")

        print("5. Exit")

        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            song_input = input("Enter song title: ").strip()
            artist_input = input("Enter artist name: ").strip()

            try:
                matched_song = get_song(song_input, artist_input, data_file)
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

            try:
                rec_songs = recommender.find_similar_songs(matched_song, n=num_recs)
                print("\nRecommended similar songs:")
                for idx, song in enumerate(rec_songs, 1):
                    print(f"{idx}. {song}")
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "2":
            print("\n--- Custom Feature Weights Recommendation ---")
            song_input = input("Enter song title: ").strip()
            artist_input = input("Enter artist name: ").strip()

            try:
                matched_song = get_song(song_input, artist_input, data_file)
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

            preferences = recommender.get_setting_preferences(matched_song)
            custom_weights = customize_weights()
            try:
                rec_songs = recommender.generate_similarity_list(matched_song, n=num_recs, preferences=preferences,
                                                                 weights=custom_weights)
                print("\nRecommended similar songs with custom weights:")
                for idx, song in enumerate(rec_songs, 1):
                    print(f"{idx}. {song}")
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "3":
            song_name = input("Enter the song title for feature analysis: ").strip()
            try:
                recommender.visualize_song_features(song_name)
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "4":
            song_name = input("Enter the song title to generate the similarity network: ").strip()
            try:
                recommender.visualize_similar_songs([song_name])
            except ValueError as e:
                print(f"Error: {e}")

        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid option, please try again.")


if __name__ == "__main__":
    main()
