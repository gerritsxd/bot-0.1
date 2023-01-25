import random
import pathlib
from schnapsen.bots import SchnapsenServer, MLPlayingBot, OkiPlayingBot
from schnapsen.bots import RandBot, AlphaBetaBot, RdeepBot
from schnapsen.game import SchnapsenGamePlayEngine, Bot
import click


@click.group()
def main() -> None:
    """Play single or multiple Schnapsen games in a GUI."""


@main.command()
@click.option('--bot', '-b',
              type=click.Choice(['AlphaBetaBot', 'RdeepBot', 'MLDataBot', 'MLPlayingBot', 'RandBot'], case_sensitive=False),
              default='RandBot', help="The bot you want to play against.")
def single(bot: str) -> None:
    """Run a single game in the GUI."""
    engine = SchnapsenGamePlayEngine()
    bot1: Bot
    with SchnapsenServer() as s:
        if bot.lower() == "randbot":
            bot1 = RandBot(12)
        elif bot.lower() in ["alphabeta", "alphabetabot"]:
            bot1 = AlphaBetaBot()
        elif bot.lower() == "rdeepbot":
            bot1 = RdeepBot(num_samples=4, depth=4, rand=random.Random(3589))
        else:
            raise NotImplementedError
        bot2 = s.make_gui_bot(name="mybot2")
        # bot1 = s.make_gui_bot(name="mybot1")
        engine.play_game(bot1, bot2, random.Random(100))


@main.command()
def multiple() -> None:
    """Run 30 games vs. ML bots in the GUI."""
    engine: SchnapsenGamePlayEngine = SchnapsenGamePlayEngine()

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
        for i in range(30):
            bot1: Bot = random.choice(bot_options)

            if bot1 == ml_bot:
                ml_bot_count += 1
                print(f'ml_bot_count has been updated to: {ml_bot_count}')
            elif bot1 == okibot:
                okibot_count += 1
                print(f'okibot_count has been updated to: {okibot_count}')

            if ml_bot_count == 15:
                ml_bot_count -= 15
                bot_options.remove(ml_bot)
                print(f'bot_options has been updated to: {bot_options}')

            if okibot_count == 15:
                okibot_count -= 15
                bot_options.remove(okibot)
                print(f'bot_options has been updated to: {bot_options}')

            bot2 = s.make_gui_bot(name=f"UNKNOWN ML BOT [GAME {i + 1}]")

            winner, points, score = engine.play_game(
                bot1,
                bot2,
                random.Random(random.randint(1, 999999))
            )

            print(
                f'Game: {i + 1}\n'
                f'Winner: {winner}\n'
                f'Points: {points}\n'
                f'Score: {score}\n'
            )


if __name__ == "__main__":
    main()
