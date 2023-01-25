import random
import pathlib
from schnapsen.bots import SchnapsenServer
from schnapsen.bots import RandBot, AlphaBetaBot, RdeepBot, MLPlayingBot
from schnapsen.game import SchnapsenGamePlayEngine, Bot
import click


@click.command()
@click.option('--bot', '-b',
              type=click.Choice(['AlphaBetaBot', 'RdeepBot', 'MLDataBot', 'MLPlayingBot', 'RandBot'], case_sensitive=False),
              default='MLPlayingBot', help="The bot you want to play against.")
@click.option('--loop', '-l',
              type=click.Choice(['single', 'loop'], case_sensitive=False),
              default='loop', help="If you want to play a single game or a loop of 30 games.")
def main(bot: str, loop: str) -> None:
    """Run the GUI."""
    engine = SchnapsenGamePlayEngine()
    bot1: Bot
    with SchnapsenServer() as s:
        if loop.lower() == 'single':
            if bot.lower() == "randbot":
                bot1 = RandBot(12)
            elif bot.lower() in ["alphabeta", "alphabetabot"]:
                bot1 = AlphaBetaBot()
            elif bot.lower() == "rdeepbot":
                bot1 = RdeepBot(num_samples=4, depth=4, rand=random.Random(3589))
            elif bot.lower() == "mlplayingbot":
                model_dir: str = 'ML_models'
                model_name: str = 'okibot_1_0_12_rand_rand_100k_games_NN_model'
                model_location = pathlib.Path(model_dir) / model_name
                bot1: Bot = MLPlayingBot(model_location=model_location)
            else:
                raise NotImplementedError
            bot2 = s.make_gui_bot(name="mybot2")
            # bot1 = s.make_gui_bot(name="mybot1")
            engine.play_game(bot1, bot2, random.Random(random.randint(0, 1000000)))
        else:
            ml_bot_name: str = 'ml_bot_rand_rand_100k_games_NN_model'
            okibot_name: str = 'okibot_1_0_12_rand_rand_100k_games_NN_model'

            options: List[str] = [
                ml_bot_name,
                okibot_name
            ]

            ml_bot_count: int = 0
            okibot_count: int = 0

            for _ in range(30):
                if bot.lower() == "mlplayingbot":
                    model_dir: str = 'ML_models'
                    model_name: str = random.choice(options)
                    model_location = pathlib.Path(model_dir) / model_name
                    bot1: Bot = MLPlayingBot(model_location=model_location)

                    if model_name == ml_bot_name:
                        ml_bot_count += 1
                    elif model_name == okibot_name:
                        okibot_count += 1

                    if ml_bot_count == 15:
                        options.remove(ml_bot_name)

                    if okibot_count == 15:
                        options.remove(okibot_name)
                else:
                    raise NotImplementedError

                bot2: Bot = s.make_gui_bot(name="mybot2")
                engine.play_game(bot1, bot2, random.Random(random.randint(1, 1000000)))


if __name__ == "__main__":
    main()
