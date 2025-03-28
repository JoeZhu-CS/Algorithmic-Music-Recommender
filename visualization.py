import networkx as nx
from plotly.graph_objs import Scatterpolar, Scatter, Figure, Bar
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

# Import your data loading function
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

    def find_similar_songs(self, song_name: str, n: int = 10) -> List[str]:
        """Find songs similar to the given song.

        Args:
            song_name: Name of the reference song
            n: Number of similar songs to return

        Returns:
            List of similar song names

        >>> recommender = MusicRecommender('SpotifySongs_no_id.csv')
        >>> similar_songs = recommender.find_similar_songs("Shape of You", 5)
        >>> len(similar_songs) <= 5
        True
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"Song '{song_name}' not found in the dataset")

        # Calculate similarities
        similarities = self._calculate_similarity(song_name)

        # Sort by similarity score (highest first)
        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        # Return top N song names
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
                                output_file: str = '') -> None:
        """Visualize a network of songs similar to the input songs.

        Args:
            song_names: List of input song names
            max_similar: Maximum number of similar songs to show per input song
            threshold: Minimum similarity score to include a song
            output_file: Path to save the visualization (empty to display in browser)
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

    # Example: Find similar songs to "Shape of You"
    song_title = "Shape of You"
    try:
        similar_songs = recommender.find_similar_songs(song_title, 10)

        print(f"Songs similar to '{song_title}':")
        import pprint

        pprint.pprint(similar_songs)

        # Visualize the similar songs
        recommender.visualize_similar_songs([song_title])

        # Visualize the features of the song
        recommender.visualize_song_features(song_title)

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

        print("Showing feature weights visualization")
        recommender.visualize_feature_weights(weights)

    except ValueError as e:
        print(f"Error: {e}")
        print("Try another song like 'Despacito' or 'Closer'")
