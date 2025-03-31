import networkx as nx
from plotly.graph_objs import Scatterpolar, Scatter, Figure, Bar
from typing import Dict, List, Optional
import os
from custom_weight_selector import customize_weights
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
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same artist? \nInput Yes or No \n> ").strip().lower()
        same_artist = {"yes": True, "no": False, "y": True, "n": False}[answer]

        # === Same Genre ===
        answer = input(
            "Restrict songs to those from the same genre (estimated genre from playlist information)? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same genre? \nInput Yes or No \n> ").strip().lower()
        same_genre = {"yes": True, "no": False, "y": True, "n": False}[answer]

        # === Same Key ===
        answer = input("Restrict songs to those of the same key? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same key? \nInput Yes or No \n> ").strip().lower()
        same_key = {"yes": True, "no": False, "y": True, "n": False}[answer]

        # === Same Mode ===
        answer = input("Restrict songs to those of the same mode (Major/Minor)? \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Same mode? \nInput Yes or No \n> ").strip().lower()
        same_mode = {"yes": True, "no": False, "y": True, "n": False}[answer]

        # === Restrict Tempo ===
        answer = input(
            f"Restrict songs based on tempo? For reference, your chosen song, \"{song_name}\", has a tempo of {song[1][11]} bpm. \nInput Yes or No \n> ").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Restrict tempo? \nInput Yes or No \n> ").strip().lower()
        if {"yes": True, "no": False, "y": True, "n": False}[answer]:
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
        while answer not in ["yes", "no", "y", "n"]:
            print("Invalid Answer. Please try again.")
            answer = input("Restrict duration? \nInput Yes or No \n> ").strip().lower()
        if {"yes": True, "no": False, "y": True, "n": False}[answer]:
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

        # Default weights if none provided - prioritize valence
        if weights is None:
            weights = [0.05, 0.12, 0.12, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.05, 0.15, 0.10, 0.05]

        # Store original song data in variables for faster access
        original_song = self.song_dict[song_name.lower()]
        original_song_data = original_song[1]

        # Set defaults to avoid uninitialized variables
        same_artist = same_genre = same_key = same_mode = False
        lower_bound_tempo = lower_bound_duration = float('-inf')
        upper_bound_tempo = upper_bound_duration = float('inf')

        og_artist = og_genre = None
        og_key = og_mode = None

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
                # Don't normalize tempo - use direct difference
                elif i == 11:
                    normalized_diff = abs(feat1 - feat2) / float('inf')  # Set to infinity to minimize impact
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

        # Otherwise, use default weights without filtering
        return self.generate_similarity_list(song_name, n)

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

        # Use preferences for similarity calculation
        all_similar_songs = {}
        preferences_to_use = None

        for song in valid_songs:
            if use_preferences:
                # Use existing preferences if provided, otherwise collect new ones
                if existing_preferences:
                    preferences_to_use = existing_preferences
                    print(f"Using your previous preferences for '{song}' recommendations.")
                else:
                    print(f"Let's collect your preferences for '{song}' recommendations.")
                    preferences_to_use = self.get_setting_preferences(song)

            # Get similar songs using generate_similarity_list with or without preferences
            similar_songs = self.generate_similarity_list(song, max_similar * 3, preferences_to_use)

            # Calculate similarity scores for visualization purposes
            similarity_scores = {}
            for similar_song in similar_songs:
                # For simplicity, we'll use inverse of position as similarity score
                # Higher position = lower similarity
                pos = similar_songs.index(similar_song)
                similarity_scores[similar_song] = 1 - (pos / (len(similar_songs) + 1))

            all_similar_songs[song.lower()] = similarity_scores

        # Create and visualize the graph
        graph = self._create_song_graph(valid_songs, all_similar_songs,
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
                marker=dict(color='rgb(55, 83, 109)')
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
