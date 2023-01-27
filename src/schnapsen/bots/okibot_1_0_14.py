from schnapsen.game import Bot, PlayerPerspective,\
    SchnapsenDeckGenerator, Move, Trick, GamePhase
from typing import List, Optional, cast, Literal
from schnapsen.deck import Suit, Rank, CardCollection
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import joblib
import time
import pathlib


class OkiPlayingBot(Bot):
    """
    This class loads a trained ML model and uses it to play
    """

    def __init__(self, model_location: Optional[pathlib.Path]) -> None:
        """
        Create a new OkiPlayingBot which uses the model stored in the
        model_location.

        :param model_location: The file containing the model.
        """
        if model_location is None:
            model_location = pathlib.Path("ML_models/") / "test_model"
        assert (
            model_location.exists(),
            f"Model could not be found at: {model_location}"
        )
        # load model
        self.__model = joblib.load(model_location)

    def get_move(
            self,
            state: 'PlayerPerspective',
            leader_move: Optional[Move]
    ) -> 'Move':
        """
        Returns the best_move that has the highest winning probability.
        """

        # get the sate feature representation
        state_representation: list[int] = get_state_feature_vector(state)

        # get the leader's move representation, even if it is None
        leader_move_representation: list[int] = get_move_feature_vector(
            leader_move,
            state
        )

        # get all my valid moves
        my_valid_moves: list[Move] = state.valid_moves()

        my_move_representations: list[list[int]] = []

        # get the feature representations for all my valid moves
        for my_move in my_valid_moves:
            my_move_representations.append(
                get_move_feature_vector(my_move, state)
            )

        # create all model inputs, for all valid moves
        action_state_representations: list[list[int]] = []

        if state.am_i_leader():

            # If you're a leader, you don't have any follower moves
            follower_move_representation = get_move_feature_vector(None)

            # Adds int representation of state (PlayerPerspective),
            # valid moves, and our follower moves (None as we're the
            # leader).
            for my_move_representation in my_move_representations:
                action_state_representations.append(
                    state_representation
                    + my_move_representation
                    + follower_move_representation
                )
        else:
            # Adds int representation of state (PlayerPerspective),
            # valid moves, and our leader moves (None as we're the
            # follower).
            for my_move_representation in my_move_representations:
                action_state_representations.append(
                    state_representation
                    + leader_move_representation
                    + my_move_representation
                )

        # Retrieves output of trained ML model containing probabilities of
        # action_state_representations
        model_output = self.__model.predict_proba(action_state_representations)

        # Retrieves winning probabilities from output of trained ML model
        winning_probabilities_of_moves = (
            [outcome_prob[1] for outcome_prob in model_output]
        )

        highest_value: float = -1

        # Finds best_move with the highest winning probability
        for index, value in enumerate(winning_probabilities_of_moves):
            if value > highest_value:
                highest_value: float = value
                best_move: Move = my_valid_moves[index]

        assert best_move is not None
        return best_move

    def __str__(self) -> str:
        """
        Returns the informal string representation of the OkiPlayingBot.
        """
        return 'okibot'


class OkiDataBot(Bot):
    """
    This class is defined to allow the creation of a training schnapsen
    bot dataset, that allows us to train a Machine Learning (ML) Bot
    Practically, it helps us record how the game plays out according to
    a provided Bot behaviour; build what is called a "replay memory" In
    more detail, we create one training sample for each decision the bot
    makes within a game, where a decision is an action selection for a
    specific game state. Then we relate each decision with the outcome
    of the game, i.e. whether this bot won or not. This way we can then
    train a bot according to the assumption that:

    "decisions in earlier games that ended up in victories should be
    preferred over decisions that lead to lost games"

    This class only records the decisions and game outcomes of the
    provided bot, according to its own perspective - incomplete game
    state knowledge.
    """

    def __init__(self, bot: Bot, replay_memory_location: pathlib.Path) -> None:
        """
        :param bot: the provided bot that will actually play the game
        and make decisions
        :param replay_memory_location: the filename under which the
        replay memory records will be
        """
        self.bot: Bot = bot
        self.replay_memory_file_path: pathlib.Path = replay_memory_location

    def get_move(
            self, state: PlayerPerspective, leader_move: Optional[Move]
    ) -> Move:
        """
        This function simply calls the get_move of the provided bot
        """
        return self.bot.get_move(state=state, leader_move=leader_move)

    def notify_game_end(self, won: bool, state: PlayerPerspective) -> None:
        """
        When the game ends, this function retrieves the game history and
        more specifically all the replay memories that can be derived
        from it, and stores them in the form of state-actions vector
        representations and the corresponding outcome of the game.

        :param won: Did this bot win the game?
        :param state: The final state of the game.
        """
        # we retrieve the game history while actually discarding the
        # last useless history record (which is after the game has
        # ended),
        # we know none of the Tricks can be None because that is only
        # for the last record
        game_history: list[tuple[PlayerPerspective, Trick]] = (
            cast(
                list[tuple[PlayerPerspective, Trick]],
                state.get_game_history()[:-1]
            )
        )

        # we also save the training label "won or lost"
        won_label = won

        # we iterate over all the rounds of the game
        for round_player_perspective, round_trick in game_history:

            if round_trick.is_trump_exchange():
                leader_move = round_trick.exchange
                follower_move = None
            else:
                leader_move = round_trick.leader_move
                follower_move = round_trick.follower_move

            # we do not want this representation to include actions that
            # followed. So if this agent was the leader, we ignore the
            # followers move
            if round_player_perspective.am_i_leader():
                follower_move = None

            state_actions_representation = (
                create_state_and_actions_vector_representation(
                    state=round_player_perspective,
                    leader_move=leader_move,
                    follower_move=follower_move
                )
            )

            # append replay memory to file
            with open(file=self.replay_memory_file_path, mode="a") as replay_memory_file:
                # replay_memory_line: List[Tuple[list, number]] = [state_actions_representation, won_label]

                # writing to replay memory file in the form:
                # "[feature list] || int(won_label)]
                replay_memory_file.write(f"{str(state_actions_representation)[1:-1]} || {int(won_label)}\n")


def train_Oki_model(replay_memory_location: Optional[pathlib.Path],
                   model_location: Optional[pathlib.Path],
                   model_class: Literal["NN", "LR"] = "LR"
                   ) -> None:
    """
    Train the ML model for the OkiPlayingBot based on replay memory
    stored by the OkiDataBot. This implementation has the option to train
    a neural network model or a model based on linear regression. The
    model classes used in this implementation are not necessarily
    optimal.

    :param replay_memory_location: Location of the games stored by
    OkiDataBot,
    default pathlib.Path('ML_replay_memories') / 'test_replay_memory'

    :param model_location: Location where the model will be stored,
    default pathlib.Path("ML_models") / 'test_model'

    :param model_class: The machine learning model class to be used,
    either 'NN' for a neural network, or 'LR' for a linear regression.

    :param overwrite: Whether to overwrite a possibly existing model.
    """
    if replay_memory_location is None:
        replay_memory_location = (
            pathlib.Path('ML_replay_memories') / 'test_replay_memory'
        )

    if model_location is None:
        model_location = pathlib.Path("ML_models") / 'test_model'

    assert model_class == 'NN' or model_class == 'LR', "Unknown model class"

    # check that the replay memory dataset is found at the specified
    # location
    if not replay_memory_location.exists():
        raise ValueError(f"Dataset not found at: {replay_memory_location} !")

    # Check if model exists already
    if model_location.exists():
        raise ValueError(
            f"Model at {model_location} exists already and overwrite is set to False. \nNo new model will be trained, process terminates"
        )

    # check if directory exists, and if not, then create it
    model_location.parent.mkdir(parents=True, exist_ok=True)

    data: list[list[int]] = []
    targets: list[int] = []

    with open(file=replay_memory_location, mode="r") as replay_memory_file:
        for line in replay_memory_file:
            feature_string, won_label_str = line.split("||")
            feature_list_strings: list[str] = feature_string.split(",")
            feature_list = [int(feature) for feature in feature_list_strings]
            won_label = int(won_label_str)
            data.append(feature_list)
            targets.append(won_label)

    print("Dataset Statistics:")
    samples_of_wins = sum(targets)
    samples_of_losses = len(targets) - samples_of_wins
    print("Samples of wins:", samples_of_wins)
    print("Samples of losses:", samples_of_losses)

    # What type of model will be used depends on the value of the
    # parameter use_neural_network
    if model_class == 'NN':
        #############################################
        # Neural Network model parameters:
        # learn more about the model or how to better use it by
        # checking out its documentation:
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier
        # Play around with the model parameters below

        print("Training a Complex (Neural Network) model.")

        # Feel free to experiment with different number of neural layers
        # or different type of neurons per layer

        # Tips: more neurons or more layers of neurons create a more
        # complicated model that takes more time to train and
        # needs a bigger dataset, but if you find the correct
        # combination of neurons and neural layers and provide a big
        # enough training dataset can lead to better performance

        # one layer of 30 neurons
        hidden_layer_sizes = (30)
        # two layers of 30 and 5 neurons respectively
        # hidden_layer_sizes = (30, 5)

        # The learning rate determines how fast we move towards the
        # optimal solution.
        # A low learning rate will converge slowly, but a large one
        # might overshoot.
        learning_rate = 0.0001

        # The regularization term aims to prevent over-fitting, and we
        # can tweak its strength here.
        regularization_strength = 0.0001

        # Train a neural network
        learner = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            alpha=regularization_strength,
            verbose=True,
            early_stopping=True,
            n_iter_no_change=6,
            activation='tanh'
        )
    elif model_class == 'LR':

        # Train a simpler Linear Logistic Regression model
        # learn more about the model or how to use better use it by
        # checking out its documentation:
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression

        print("Training a Simple (Linear Logistic Regression model)")

        # Usually there is no reason to change the hyperparameters of
        # such a simple model but fill free to experiment:
        learner = LogisticRegression(max_iter=1250)
    else:
        raise AssertionError("Unknown model class")

    start = time.time()
    print("Starting training phase...")

    model = learner.fit(data, targets)
    # Save the model in a file
    joblib.dump(model, model_location)
    end = time.time()
    print('The model was trained in ', (end - start) / 60, 'minutes.')


def create_state_and_actions_vector_representation(
        state: PlayerPerspective,
        leader_move: Optional[Move],
        follower_move: Optional[Move]
) -> List[int]:
    """
    This function takes as input a PlayerPerspective variable, and the
    two moves of leader and follower, and returns a list of complete
    feature representation that contains all information.
    """
    player_game_state_representation = get_state_feature_vector(state)
    leader_move_representation = get_move_feature_vector(leader_move, state)
    follower_move_representation = get_move_feature_vector(
        follower_move, state
    )

    return (
        player_game_state_representation
        + leader_move_representation
        + follower_move_representation
    )


def get_one_hot_encoding_of_card_suit(card_suit: Suit) -> List[int]:
    """
    Translating the suit of a card into 1-hot vector encoding of size
    4.
    """
    card_suit_one_hot: list[int]
    if card_suit == Suit.HEARTS:
        card_suit_one_hot = [0, 0, 0, 1]
    elif card_suit == Suit.CLUBS:
        card_suit_one_hot = [0, 0, 1, 0]
    elif card_suit == Suit.SPADES:
        card_suit_one_hot = [0, 1, 0, 0]
    elif card_suit == Suit.DIAMONDS:
        card_suit_one_hot = [1, 0, 0, 0]
    else:
        raise ValueError("Suit of card was not found!")

    return card_suit_one_hot


def get_one_hot_encoding_of_card_rank(card_rank: Rank) -> List[int]:
    """
    Translating the rank of a card into 1-hot vector encoding of size
    13.
    """
    card_rank_one_hot: list[int]
    if card_rank == Rank.ACE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif card_rank == Rank.TWO:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif card_rank == Rank.THREE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif card_rank == Rank.FOUR:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif card_rank == Rank.FIVE:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif card_rank == Rank.SIX:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif card_rank == Rank.SEVEN:
        card_rank_one_hot = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.EIGHT:
        card_rank_one_hot = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.NINE:
        card_rank_one_hot = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.TEN:
        card_rank_one_hot = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.JACK:
        card_rank_one_hot = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.QUEEN:
        card_rank_one_hot = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    elif card_rank == Rank.KING:
        card_rank_one_hot = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        raise AssertionError("Provided card Rank does not exist!")
    return card_rank_one_hot


def get_move_feature_vector(
        move: Optional[Move],
        state: PlayerPerspective = None
) -> List[int]:
    """
    In case there isn't any move provided move to encode, we still need
    to create a "padding"-"meaningless" vector of the same size, filled
    with 0s, since the ML models need to receive input of the same
    dimensionality always. Otherwise, we create all the information of
    the move i) move type, ii) played card rank and iii) played card
    suit translate this information into one-hot vectors respectively,
    and concatenate these vectors into one move feature representation
    vector.
    """

    if move is None:
        move_type_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_rank_one_hot_encoding_numpy_array = [0, 0, 0, 0]
        card_suit_one_hot_encoding_numpy_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    else:
        move_type_one_hot_encoding: list[int]

        # a) in case the move is a royal marriage move
        if move.is_marriage() and move.suit == state.get_trump_suit():
            move_type_one_hot_encoding = [0, 0, 0, 1]
            card = move.queen_card

        # b) in case the move is a marriage move
        if move.is_marriage():
            move_type_one_hot_encoding = [0, 0, 1, 0]
            card = move.queen_card

        # c) in case the move is a trump exchange move
        elif move.is_trump_exchange():
            move_type_one_hot_encoding = [0, 1, 0, 0]
            card = move.jack

        # d) in case it is a regular move
        else:
            move_type_one_hot_encoding = [1, 0, 0, 0]
            card = move.card

        move_type_one_hot_encoding_numpy_array = move_type_one_hot_encoding

        card_rank_one_hot_encoding_numpy_array = (
            get_one_hot_encoding_of_card_rank(card.rank)
        )

        card_suit_one_hot_encoding_numpy_array = (
            get_one_hot_encoding_of_card_suit(card.suit)
        )

    return (
        move_type_one_hot_encoding_numpy_array
        + card_rank_one_hot_encoding_numpy_array
        + card_suit_one_hot_encoding_numpy_array
    )


def get_state_feature_vector(state: PlayerPerspective) -> List[int]:
    """
    Accepts the player's perspective, gathers all subjective information
    accessible by this bot, adds this to the state feature set, and
    returns this state feature set, so it can be used to decide the
    OkiPlayingBot's next move.

    Features added:
    - player's points (int)
    - opponent's points (int)
    - player's pending points (int)
    - opponent's pending points (int)
    - player's total points (int)
    - opponent's total points (int)
    - trump suit (1-hot encoding)
    - game phase (1-hot encoding)
    - talon size (int)
    - whether the player is the leader (1-hot encoding)
    - position of each card of the deck (1-hot encoding)
    - number of hand cards (int)
    - number of cards won by the player (int)
    - number of known opponent cards (int)
    - number of cards won by the opponent (int)
    - number of unknown cards of the trump suit (int)
    - number of unknown aces (int)
    - number of unknown tens (int)
    - number of unknown kings (int)
    - number of unknown queens (int)
    - number of unknown jacks (int)
    - number of seen cards of the trump suit (int)
    - number of seen aces (int)
    - number of seen tens (int)
    - number of seen kings (int)
    - number of seen queens (int)
    - number of seen jacks (int)

    IMPORTANT: This function should NOT include the move of this
    agent. It should ONLY include any earlier actions of other
    agents (so the action of the other agent in case that is the
    leader)
    """
    # Initialize a list of all the features that comprise the state
    # feature set (of type np.ndarray).
    state_feature_list: list[int] = []

    # Get the player's score (including direct & pending points).
    player_score = state.get_my_score()

    # Get the player's points.
    player_points = player_score.direct_points

    # Get the player's pending points.
    player_pending_points = player_score.pending_points

    # Get the player's total points.
    player_tot_points: int = player_points + player_pending_points

    # Get the opponent's score (including direct & pending points).
    opponents_score = state.get_opponent_score()

    # Get the opponent's points.
    opponents_points = opponents_score.direct_points

    # Get the opponent's pending points.
    opponents_pending_points = opponents_score.pending_points

    # Get the opponent's total points.
    opponents_tot_points: int = opponents_points + opponents_pending_points

    # Get the trump suit (1-hot encoding).
    trump_suit = state.get_trump_suit()
    trump_suit_one_hot = get_one_hot_encoding_of_card_suit(trump_suit)

    # Get the game phase (1-hot encoding).
    game_phase_encoded = [1, 0] if state.get_phase() == GamePhase.TWO else [0, 1]

    # Get the number of remaining cards in the talon.
    talon_size: int = state.get_talon_size()

    # Determine whether the player is a leader (1-hot encoding).
    i_am_leader = [0, 1] if state.am_i_leader() else [1, 0]

    # Gather all known deck information.
    hand_cards = state.get_hand().cards
    trump_card = state.get_trump_card()
    won_cards = state.get_won_cards().get_cards()
    opponent_won_cards = state.get_opponent_won_cards().get_cards()
    opponent_known_cards = state.get_known_cards_of_opponent_hand().get_cards()

    # Each card can either be:
    # i)   in player's hand,
    # ii)  in player's won cards,
    # iii) in opponent's hand,
    # iv)  in opponent's won cards,
    # v)   be the trump card, or
    # vi)  in an unknown position
    #      (either in the talon or in the opponent's hand).

    # These are six different cases regarding knowledge of a card's
    # position, represented using 1-hot encoding vectors.
    deck_knowledge_in_consecutive_one_hot_encodings: list[int] = []

    for card in SchnapsenDeckGenerator().get_initial_deck():
        card_knowledge_in_one_hot_encoding: list[int]

        # i) in player's hand
        if card in hand_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 0, 1]

        # ii) in player's won cards
        elif card in won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 0, 1, 0]

        # iii) in opponent's hand
        elif card in opponent_known_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 0, 1, 0, 0]

        # iv) in opponent's won cards
        elif card in opponent_won_cards:
            card_knowledge_in_one_hot_encoding = [0, 0, 1, 0, 0, 0]

        # v) be the trump card
        elif card == trump_card:
            card_knowledge_in_one_hot_encoding = [0, 1, 0, 0, 0, 0]

        # vi) in an unknown position as it is invisible to this player.
        #     Thus, it is either in the talon or in the opponent's hand.
        else:
            card_knowledge_in_one_hot_encoding = [1, 0, 0, 0, 0, 0]

        # This list eventually develops to one long 1-dimensional numpy
        # array of shape (120,)
        deck_knowledge_in_consecutive_one_hot_encodings += card_knowledge_in_one_hot_encoding

    #deck_knowledge_flattened: np.ndarray = np.concatenate(tuple(deck_knowledge_in_one_hot_encoding), axis=0)

    # Get the number of cards in the player's hand.
    num_hand_cards: int = len(hand_cards)

    # Get the number of cards won by the player.
    num_won_cards: int = len(won_cards)

    # Get the number of known cards in the opponent's hand.
    num_opp_known_cards: int = len(opponent_known_cards)

    # Get the number of cards won by the opponent.
    num_opp_won_cards: int = len(opponent_won_cards)

    # Create the list of seen cards.
    seen_cards = (
            hand_cards
            + [trump_card]
            + won_cards
            + opponent_won_cards
            + opponent_known_cards
    )

    # Initialize the number of unknown cards.
    unknown_trump_cards: int = 5
    unknown_aces: int = 4
    unknown_tens: int = 4
    unknown_kings: int = 4
    unknown_queens: int = 4
    unknown_jacks: int = 4

    # Count the number of unknown cards.
    for card in seen_cards:
        if card:
            if card.suit == trump_suit:
                unknown_trump_cards -= 1

            if card.rank == Rank.ACE:
                unknown_aces -= 1
            elif card.rank == Rank.TEN:
                unknown_tens -= 1
            elif card.rank == Rank.KING:
                unknown_kings -= 1
            elif card.rank == Rank.QUEEN:
                unknown_queens -= 1
            elif card.rank == Rank.JACK:
                unknown_jacks -= 1

    # Compute the number of seen cards.
    seen_trump_cards: int = 5 - unknown_trump_cards
    seen_aces: int = 4 - unknown_aces
    seen_tens: int = 4 - unknown_tens
    seen_kings: int = 4 - unknown_kings
    seen_queens: int = 4 - unknown_queens
    seen_jacks: int = 4 - unknown_jacks

    # Add the features to the feature set.
    state_feature_list += [player_points]
    state_feature_list += [player_pending_points]
    state_feature_list += [player_tot_points]
    state_feature_list += [opponents_points]
    state_feature_list += [opponents_pending_points]
    state_feature_list += [opponents_tot_points]
    state_feature_list += trump_suit_one_hot
    state_feature_list += game_phase_encoded
    state_feature_list += [talon_size]
    state_feature_list += i_am_leader
    state_feature_list += deck_knowledge_in_consecutive_one_hot_encodings
    state_feature_list += [num_hand_cards]
    state_feature_list += [num_won_cards]
    state_feature_list += [num_opp_known_cards]
    state_feature_list += [num_opp_won_cards]
    state_feature_list += [unknown_trump_cards]
    state_feature_list += [unknown_aces]
    state_feature_list += [unknown_tens]
    state_feature_list += [unknown_kings]
    state_feature_list += [unknown_queens]
    state_feature_list += [unknown_jacks]
    state_feature_list += [seen_trump_cards]
    state_feature_list += [seen_aces]
    state_feature_list += [seen_tens]
    state_feature_list += [seen_kings]
    state_feature_list += [seen_queens]
    state_feature_list += [seen_jacks]

    return state_feature_list
