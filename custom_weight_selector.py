def customize_weights(default_weights=None, max_selection=5, high_value=0.15, low_value=0.05):

    """
    Allows the user to select which feature weights to emphasize based on the parameter number.

    By default, if the user does not enter any number, the original default_weights are returned:

    If the user enters a parameter number (up to max_selection), selected parameters are assigned high_value,
    unselected parameters are assigned low_value, and unselected parameters are assigned low_value.

    Unchecked parameters are given a low_value and returned normalized.
    """
    param_names = [
        "Popularity", "Danceability", "Energy", "Key", "Loudness",
        "Mode", "Speechiness", "Acousticness", "Instrumentalness",
        "Liveness", "Valence", "Tempo", "Duration"
    ]

    if default_weights is None:
        default_weights = [0.05, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05]

    print("\nDefault feature weights:")
    for i, (name, weight) in enumerate(zip(param_names, default_weights), start=1):
        print(f"{i}. {name}: {weight}")

    indices_input = input(
        f"\nEnter the indices (comma separated) of parameters that you want to give higher priority \n"
        f"in the recommendation process (max {max_selection}).\n"
        "Press Enter to keep default weights: "
    ).strip()

    if not indices_input:
        return default_weights

    try:
        indices = [int(x.strip()) for x in indices_input.split(",") if x.strip()]
    except ValueError:
        print("Invalid input. Using default weights.")
        return default_weights

    valid_indices = []
    for idx in indices:
        if 1 <= idx <= len(param_names) and idx not in valid_indices:
            valid_indices.append(idx)
    if len(valid_indices) > max_selection:
        print(
            f"You can only emphasize up to {max_selection} parameters. Taking first {max_selection} valid selections.")
        valid_indices = valid_indices[:max_selection]

    new_raw_weights = []
    for i in range(len(param_names)):
        if (i + 1) in valid_indices:
            new_raw_weights.append(high_value)
        else:
            new_raw_weights.append(low_value)

    total = sum(new_raw_weights)
    normalized_weights = [w / total for w in new_raw_weights]

    print("\nCustomized feature weights (normalized):")
    for i, (name, weight) in enumerate(zip(param_names, normalized_weights), start=1):
        print(f"{i}. {name}: {weight:.4f}")

    return normalized_weights
