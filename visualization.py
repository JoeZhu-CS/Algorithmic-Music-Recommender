import networkx as nx
from plotly.graph_objs import Scatterpolar, Scatter, Figure, Bar
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# Import data loading function
from get_dict_from_data import get_dict_from_data

# Colours to use when visualizing different clusters of songs
COLOUR_SCHEME = [
    '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100',
    '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D',
    '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA',
    '#6C4516', '#0D2A63', '#AF0038'
]

LINE_COLOUR = 'rgb(210,210,210)'
VERTEX_BORDER_COLOUR = 'rgb(50, 50, 50)'
SONG_COLOUR = 'rgb(89, 205, 105)'
INPUT_SONG_COLOUR = 'rgb(105, 89, 205)'


class MusicRecommender:
    """A class for recommending and visualizing similar songs."""

    def __init__(self, song_data_path: str):
        """Initialize the recommender with song data.

        Args:
            song_data_path: Path to the CSV file containing song data
        """
        self.song_dict = get_dict_from_data(song_data_path)

    def get_setting_preferences(self, song_name: str) -> tuple[bool, bool, bool, bool, float, float, float, float]:
        """
        Returns the user's preferred settings for the search. Takes in a song name.

        Args:
            song_name: Name of the reference song

        Returns:
            Tuple containing:
            - same_artist: Whether to restrict to same artist
            - same_genre: Whether to restrict to same genre
            - same_key: Whether to restrict to same key
            - same_mode: Whether to restrict to same mode
            - lower_bound_tempo: Minimum tempo
            - upper_bound_tempo: Maximum tempo
            - lower_bound_duration: Minimum duration
            - upper_bound_duration: Maximum duration
        """
        # Get the song data
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

        song = self.song_dict[song_name.lower()]

        # === Same Artist ===
        answer = input("Restrict songs to those from the same artist? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same artist? \nInput Yes or No \n> ").strip().lower()
        same_artist = {"yes": True, "no": False}[answer]

        # === Same Genre ===
        answer = input(
            "Restrict songs to those from the same genre (estimated genre from playlist information)? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same genre? \nInput Yes or No \n> ").strip().lower()
        same_genre = {"yes": True, "no": False}[answer]

        # === Same Key ===
        answer = input("Restrict songs to those of the same key? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same key? \nInput Yes or No \n> ").strip().lower()
        same_key = {"yes": True, "no": False}[answer]

        # === Same Mode ===
        answer = input("Restrict songs to those of the same mode (Major/Minor)? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same mode? \nInput Yes or No \n> ").strip().lower()
        same_mode = {"yes": True, "no": False}[answer]

        # === Restrict Tempo ===
        answer = input(
            f"Restrict songs based on tempo? For reference, your chosen song, \"{song_name}\", has a tempo of {song[1][11]} bpm. \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Restrict tempo? \nInput Yes or No \n> ").strip().lower()
        if {"yes": True, "no": False}[answer]:
            answer = input("Input a lower bound for tempo (bpm). (integers only) \n> ").strip()
            while not answer.isdigit():
                print("Invalid Answer. Please try again.")
                answer = input("Input a lower bound for tempo (bpm). \n> ").strip()
            lower_bound_tempo = float(answer)

            answer = input("Input an upper bound for tempo (bpm). (integers only) \n> ").strip()
            while not answer.isdigit():
                print("Invalid Answer. Please try again.")
                answer = input("Input an upper bound for tempo (bpm). \n> ").strip()
            upper_bound_tempo = float(answer)
        else:
            lower_bound_tempo = 0
            upper_bound_tempo = float('inf')

        # === Restrict Duration ===
        answer = input(
            f"Restrict songs based on song duration? For reference, your chosen song, \"{song_name}\", has a duration of {int(song[1][12] / 1000)} seconds, or around {round(song[1][12] / 1000 / 60, 3)} minutes. \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no"]:
            print("Invalid Answer. Please try again.")
            answer = input("Restrict duration? \nInput Yes or No \n> ").strip().lower()
        if {"yes": True, "no": False}[answer]:
            answer = input("Input a lower bound for duration (seconds). (integers only) \n> ").strip()
            while not answer.isdigit():
                print("Invalid Answer. Please try again.")
                answer = input("Input a lower bound for duration (seconds). \n> ").strip()
            lower_bound_duration = float(answer) * 1000

            answer = input("Input an upper bound for duration (seconds). (integers only) \n> ").strip()
            while not answer.isdigit():
                print("Invalid Answer. Please try again.")
                answer = input("Input an upper bound for duration (seconds). \n> ").strip()
            upper_bound_duration = float(answer) * 1000
        else:
            lower_bound_duration = 0
            upper_bound_duration = float('inf')

        return same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration

    def generate_similarity_list(self, song_name: str, n: int = 10,
                                 preferences: Optional[
                                     tuple[bool, bool, bool, bool, float, float, float, float]] = None,
                                 weights: Optional[List[float]] = None) -> List[str]:
        """
        Generate a list of similar songs based on user preferences and feature weights.

        Args:
            song_name: Name of the reference song
            n: Number of similar songs to return
            preferences: Tuple of filtering preferences (same_artist, same_genre, same_key, etc.)
            weights: List of weights for different features in similarity calculation
                    [popularity, danceability, energy, key, loudness, mode, speechiness,
                     acousticness, instrumentalness, liveness, valence, tempo, duration]

        Returns:
            List of similar song names
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

        # Default weights if none provided
        if weights is None:
            weights = [0.05, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05]

        # Store original song data in variables for faster access
        original_song = self.song_dict[song_name.lower()]
        original_song_data = original_song[1]

        if preferences:
            # Unpack user preferences
            same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration = preferences

            # Get reference values for filtering
            og_artist = original_song[0][0]
            og_genre = original_song[0][1]
            og_key = original_song_data[3]
            og_mode = original_song_data[5]

        # Dictionary to store song similarities
        similarities = {}

        # Calculate similarity for each song
        for other_song_name, other_song in self.song_dict.items():
            # Skip the reference song itself
            if other_song_name == song_name.lower():
                continue

            # Apply filters if preferences are provided
            if preferences:
                # Skip songs that don't match the filter criteria
                if (
                        (same_artist and og_artist != other_song[0][0]) or
                        (same_genre and og_genre != other_song[0][1]) or
                        (same_key and og_key != other_song[1][3]) or
                        (same_mode and og_mode != other_song[1][5]) or
                        (other_song[1][11] < lower_bound_tempo) or
                        (other_song[1][11] > upper_bound_tempo) or
                        (other_song[1][12] < lower_bound_duration) or
                        (other_song[1][12] > upper_bound_duration)
                ):
                    continue

            # Calculate weighted Euclidean distance for all numerical features
            distance = 0
            other_features = other_song[1]

            # Sum up the weighted squared differences for each feature
            for i, (feat1, feat2) in enumerate(zip(original_song_data, other_features)):
                # Normalize key (0-11)
                if i == 3:
                    normalized_diff = abs(feat1 - feat2) / 11
                    distance += weights[i] * normalized_diff ** 2
                # Normalize loudness (typically -60 to 0)
                elif i == 4:
                    normalized_diff = abs((feat1 + 60) / 60 - (feat2 + 60) / 60)
                    distance += weights[i] * normalized_diff ** 2
                # Normalize tempo (0-250)
                elif i == 11:
                    normalized_diff = abs(feat1 - feat2) / 250
                    distance += weights[i] * normalized_diff ** 2
                # Normalize duration
                elif i == 12:
                    normalized_diff = abs(feat1 - feat2) / (5 * 60 * 1000)  # Normalize to 5 minutes
                    distance += weights[i] * normalized_diff ** 2
                # Other features are already in 0-1 range
                else:
                    distance += weights[i] * (feat1 - feat2) ** 2

            # Convert distance to similarity (closer = more similar)
            similarity = 1 / (1 + distance ** 0.5)
            similarities[other_song_name] = similarity

        # Sort by similarity score (highest first)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Return top N song names
        return [song for song, _ in sorted_songs[:n]]

    def find_similar_songs(self, song_name: str, n: int = 10,
                           use_preferences: bool = False) -> List[str]:
        """Find songs similar to the given song.

        Args:
            song_name: Name of the reference song
            n: Number of similar songs to return
            use_preferences: Whether to use user preferences for filtering

        Returns:
            List of similar song names

        >>> recommender = MusicRecommender('SpotifySongs_no_id.csv')
        >>> similar_songs = recommender.find_similar_songs("Shape of You", 5)
        >>> len(similar_songs) <= 5
        True
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

        # Use preferences-based method if requested
        if use_preferences:
            print("Let's collect your preferences for song recommendations.")
            preferences = self.get_setting_preferences(song_name)
            return self.generate_similarity_list(song_name, n, preferences)

        # Otherwise, use the original method
        similarities = self._calculate_similarity(song_name)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [song for song, _ in sorted_songs[:n]]

    def _calculate_similarity(self, song_name: str) -> Dict[str, float]:
        """Calculate similarity between the given song and all other songs.

        Args:
            song_name: Name of the reference song

        Returns:
            Dictionary mapping song names to similarity scores
        """
        reference_song = self.song_dict[song_name.lower()]
        ref_features = reference_song[1]  # Numerical features

        similarities = {}
        for other_song, other_data in self.song_dict.items():
            if other_song == song_name.lower():
                continue

            other_features = other_data[1]

            # Calculate weighted Euclidean distance for key features
            distance = (
                               0.3 * (ref_features[1] - other_features[1]) ** 2 +  # danceability
                               0.3 * (ref_features[2] - other_features[2]) ** 2 +  # energy
                               0.1 * (ref_features[7] - other_features[7]) ** 2 +  # acousticness
                               0.1 * (ref_features[8] - other_features[8]) ** 2 +  # instrumentalness
                               0.2 * (ref_features[10] - other_features[10]) ** 2  # valence
                       ) ** 0.5

            # Convert distance to similarity (closer = more similar)
            similarity = 1 / (1 + distance)
            similarities[other_song] = similarity

        return similarities

    def visualize_similar_songs(self, song_names: List[str],
                                max_similar: int = 5,
                                threshold: float = 0.7,
                                output_file: str = '',
                                use_preferences: bool = False,
                                existing_preferences: Optional[tuple] = None) -> None:
        """Visualize a network of songs similar to the input songs.

        Args:
            song_names: List of input song names
            max_similar: Maximum number of similar songs to show per input song
            threshold: Minimum similarity score to include a song
            output_file: Path to save the visualization (empty to display in browser)
            use_preferences: Whether to use user preferences for filtering
            existing_preferences: Previously collected preferences to reuse
        """
        # Validate song names
        valid_songs = []
        for song in song_names:
            if song.lower() in self.song_dict:
                valid_songs.append(song)
            else:
                print(f"Warning: Song '{song}' not found in dataset")

        if not valid_songs:
            raise ValueError("No valid songs provided")

        # Calculate similarities for each input song
        similarity_scores = {}

        for song in valid_songs:
            if use_preferences:
                # Use existing preferences if provided, otherwise collect new ones
                if existing_preferences:
                    preferences = existing_preferences
                    print(f"Using your previous preferences for '{song}' recommendations.")
                else:
                    print(f"Let's collect your preferences for '{song}' recommendations.")
                    preferences = self.get_setting_preferences(song)

                # Use generate_similarity_list, but we need to convert the result to a dictionary with scores
                top_songs = self.generate_similarity_list(song, max_similar * 3, preferences)  # Get more than needed

                # Recalculate similarities just for these songs for visualization
                temp_similarities = self._calculate_similarity(song)
                similarity_scores[song.lower()] = {s: temp_similarities[s] for s in top_songs if s in temp_similarities}
            else:
                similarity_scores[song.lower()] = self._calculate_similarity(song)

        # Create and visualize the graph
        graph = self._create_song_graph(valid_songs, similarity_scores,
                                        threshold, max_similar)
        self._visualize_song_graph(graph, output_file=output_file)

    def visualize_song_features(self, song_name: str, output_file: str = '') -> None:
        """Visualize the audio features of a specific song.

        Args:
            song_name: Name of the song to visualize
            output_file: Path to save the visualization (empty to display in browser)
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

        # Get song data
        song_data = self.song_dict[song_name.lower()]
        num_data = song_data[1]

        # Feature names (excluding popularity and duration)
        features = [
            'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
            'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo'
        ]

        # Normalize values to 0-1 range
        normalized_values = [
            num_data[1],  # danceability (already 0-1)
            num_data[2],  # energy (already 0-1)
            num_data[3] / 11,  # key (0-11)
            (num_data[4] + 60) / 60,  # loudness (typically -60 to 0)
            num_data[5],  # mode (already 0-1)
            num_data[6],  # speechiness (already 0-1)
            num_data[7],  # acousticness (already 0-1)
            num_data[8],  # instrumentalness (already 0-1)
            num_data[9],  # liveness (already 0-1)
            num_data[10],  # valence (already 0-1)
            num_data[11] / 250  # tempo (normalized to 0-250)
        ]

        # Close the loop for the radar chart
        features.append(features[0])
        normalized_values.append(normalized_values[0])

        # Create radar chart
        fig = Figure()

        fig.add_trace(Scatterpolar(
            r=normalized_values,
            theta=features,
            fill='toself',
            name=song_name
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"Feature Analysis: {song_name} - {song_data[0][0]}",
            title_x=0.5
        )

        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)

    def _create_song_graph(self, input_songs: List[str],
                           similarity_scores: Dict[str, Dict[str, float]],
                           threshold: float = 0.7,
                           max_connections: int = 5) -> nx.Graph:
        """Create a NetworkX graph from song similarity data."""
        G = nx.Graph()

        # Add input songs as nodes
        for song in input_songs:
            if song.lower() in self.song_dict:
                G.add_node(song.lower(), kind='input_song',
                           artist=self.song_dict[song.lower()][0][0],
                           genre=self.song_dict[song.lower()][0][1])

        # Add similar songs and edges
        for input_song, scores in similarity_scores.items():
            # Sort by similarity score (highest first)
            sorted_songs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

            # Add top similar songs (up to max_connections)
            for similar_song, score in sorted_songs[:max_connections]:
                if score >= threshold and similar_song in self.song_dict:
                    # Add the similar song if not already in graph
                    if similar_song not in G:
                        G.add_node(similar_song, kind='song',
                                   artist=self.song_dict[similar_song][0][0],
                                   genre=self.song_dict[similar_song][0][1])

                    # Add edge with similarity score as weight
                    G.add_edge(input_song, similar_song, weight=score)

        return G

    def _visualize_song_graph(self, graph: nx.Graph,
                              layout: str = 'spring_layout',
                              max_vertices: int = 100,
                              output_file: str = '') -> None:
        """Visualize the song similarity graph."""
        # Limit the number of vertices
        if len(graph) > max_vertices:
            graph = graph.subgraph(list(graph.nodes)[:max_vertices])

        # Get positions using the specified layout
        pos = getattr(nx, layout)(graph)

        # Extract node information
        x_values = [pos[k][0] for k in graph.nodes]
        y_values = [pos[k][1] for k in graph.nodes]

        # Create labels with song name and artist
        labels = [f"{k} - {graph.nodes[k]['artist']}" for k in graph.nodes]

        # Set colors based on node kind (input song or recommended song)
        kinds = [graph.nodes[k]['kind'] for k in graph.nodes]
        colours = [INPUT_SONG_COLOUR if kind == 'input_song' else SONG_COLOUR for kind in kinds]

        # Create edge traces
        x_edges = []
        y_edges = []

        for edge in graph.edges:
            x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
            y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]

        # Create edge trace
        edge_trace = Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            name='similarities',
            line=dict(color=LINE_COLOUR, width=1),
            hoverinfo='none',
        )

        # Create node trace
        node_trace = Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name='songs',
            marker=dict(
                symbol='circle',
                size=10,
                color=colours,
                line=dict(color=VERTEX_BORDER_COLOUR, width=0.5)
            ),
            text=labels,
            hovertemplate='%{text}<br>Genre: %{customdata[0]}',
            customdata=[[graph.nodes[k]['genre']] for k in graph.nodes],
            hoverlabel={'namelength': 0}
        )

        # Create the figure
        data = [edge_trace, node_trace]
        fig = Figure(data=data)
        fig.update_layout({
            'showlegend': False,
            'title': 'Song Similarity Network',
            'title_x': 0.5,
            'hovermode': 'closest'
        })
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)

    def visualize_feature_weights(self, weights: Dict[str, float], output_file: str = '') -> None:
        """Visualize the weights of different features in the similarity calculation.

        Args:
            weights: Dictionary mapping feature names to their weights
            output_file: Path to save the visualization (empty to display in browser)
        """
        # Sort features by weight (descending)
        sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        feature_weights = [f[1] for f in sorted_features]

        # Create bar chart
        fig = Figure(data=[
            Bar(
                x=feature_names,
                y=feature_weights,
                marker_color='rgb(55, 83, 109)'
            )
        ])

        fig.update_layout(
            title='Feature Importance in Song Recommendations',
            title_x=0.5,
            xaxis=dict(
                title='Feature',
                tickangle=45
            ),
            yaxis=dict(
                title='Weight',
                range=[0, max(feature_weights) * 1.1]
            ),
            margin=dict(b=100)
        )

        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)


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

            use_preferences = input("Which mode would you prefer? (basic/advanced): ").lower().startswith('a')

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

            # Custom weights demonstration
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

                # Demonstrate custom weights
                if input("Would you like to try custom feature weights? (yes/no): ").lower().startswith('y'):
                    print("\nCustomizing weights lets you prioritize specific audio features.")
                    print("For example, we can prioritize danceability and energy for party music.")
                    custom_weights = [0.03, 0.25, 0.25, 0.03, 0.03, 0.03, 0.03, 0.05, 0.05, 0.03, 0.15, 0.05, 0.02]

                    # Reuse existing preferences or get new ones if needed
                    if preferences is None:
                        print("\nLet's set filtering preferences first:")
                        preferences = recommender.get_setting_preferences(song_title)
                    else:
                        print("Using your previously set preferences with custom weights...")

                    # Use custom weights with preferences
                    custom_songs = recommender.generate_similarity_list(
                        song_title, 10, preferences, custom_weights
                    )

                    print(f"\nTop 10 dance-oriented recommendations for '{song_title.title()}':")
                    for i, song in enumerate(custom_songs, 1):
                        similar_artist = recommender.song_dict[song][0][0]
                        print(f"{i}. {song.title()} - {similar_artist}")

            print("\nWould you like to try another song?")
            if not input("Continue? (yes/no): ").lower().startswith('y'):
                print("\nThank you for using the Music Recommendation System!")
                break

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Let's try something else.")
