from schnapsen.bots import SchnapsenServer, MLPlayingBot, OkiPlayingBot
from schnapsen.bots import RandBot, AlphaBetaBot, RdeepBot
from schnapsen.game import SchnapsenGamePlayEngine, Bot
import pathlib
import random
import click
import csv


@click.group()
def main() -> None:
    """Play single or multiple Schnapsen games in a GUI."""


@main.command()
@click.option('--bot', '-b',
              type=click.Choice(['AlphaBetaBot', 'RdeepBot', 'MLPlayingBot', 'OkiBot', 'RandBot'], case_sensitive=False),
              default='OkiBot',
              help="The bot you want to play against (e.g. MLPlayingBot).")
def single(bot: str) -> None:
    """Run a single game in the GUI."""
    engine = SchnapsenGamePlayEngine()
    bot1: Bot
    with SchnapsenServer() as s:
        model_dir: str = 'ML_models'

        if bot.lower() == "randbot":
            bot1 = RandBot(12)
        elif bot.lower() in ["alphabeta", "alphabetabot"]:
            bot1 = AlphaBetaBot()
        elif bot.lower() == "rdeepbot":
            bot1 = RdeepBot(num_samples=4, depth=4, rand=random.Random(3589))
        elif bot.lower() == "mlplayingbot":
            ml_model: str = 'ml_bot_rand_rand_100k_games_NN_model'
            ml_location = pathlib.Path(model_dir) / ml_model
            bot1 = MLPlayingBot(model_location=ml_location)
        elif bot.lower() == "okibot":
            okibot_model: str = 'okibot_1_0_12_rand_rand_100k_games_NN_model'
            okibot_location = pathlib.Path(model_dir) / okibot_model
            bot1 = OkiPlayingBot(model_location=okibot_location)
        else:
            raise NotImplementedError

        bot2 = s.make_gui_bot(name=bot)
        #bot1 = s.make_gui_bot(name="mybot1")

        engine.play_game(bot1, bot2, random.Random(random.randint(1, 999999)))


@main.command()
@click.argument(
    'player_name',
    help='The name of the human that will play 30 games (e.g. Leen).'
)
def multiple(player_name) -> None:
    """Run 30 games vs. ML bots in the GUI."""
    engine: SchnapsenGamePlayEngine = SchnapsenGamePlayEngine()

    results_dir: str = 'ML_results'
    results_file: str = 'human_results.csv'
    results_location = pathlib.Path(results_dir) / results_file

    model_dir: str = 'ML_models'

    ml_model: str = 'ml_bot_rand_rand_100k_games_NN_model'
    ml_location = pathlib.Path(model_dir) / ml_model

    okibot_model: str = 'okibot_1_0_12_rand_rand_100k_games_NN_model'
    okibot_location = pathlib.Path(model_dir) / okibot_model

    ml_bot: Bot = MLPlayingBot(model_location=ml_location)
    okibot: Bot = OkiPlayingBot(model_location=okibot_location)

    bot_options: list[Bot] = [ml_bot, okibot]

    ml_bot_count: int = 0
    okibot_count: int = 0

    with SchnapsenServer() as s:
        for i in range(1, 31):
            bot1: Bot = random.choice(bot_options)

            if bot1 == ml_bot:
                ml_bot_count += 1
            elif bot1 == okibot:
                okibot_count += 1

            if ml_bot_count == 15:
                ml_bot_count -= 15
                bot_options.remove(ml_bot)

            if okibot_count == 15:
                okibot_count -= 15
                bot_options.remove(okibot)

            bot2 = s.make_gui_bot(name=f"Unknown bot [GAME {i}]")

            winner, game_points, round_points = engine.play_game(
                bot1,
                bot2,
                random.Random(random.randint(1, 999999))
            )

            game_data: list[any] = [
                player_name,
                i,
                bot1,
                winner
            ]

            with open(results_location, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(game_data)


if __name__ == "__main__":
    main()
