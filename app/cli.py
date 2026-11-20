"""CLI: python -m app.cli --help"""
import argparse
import json
from pathlib import Path
from .data import generate_games, load_sequences, split_sequences
from .mining import mine_frequent_itemsets, save_itemsets
from .model import NGramActionModel, TransformerActionModel, evaluate, continue_pretrain_transformer

DATA = Path("data")
MODEL_PATH = DATA / "action_model.json"

def main(argv=None):
    parser = argparse.ArgumentParser(prog="game-action-predictor",
        description="Mine basketball play patterns and predict the next action.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check the installation")
    gen = sub.add_parser("prepare", help="generate reproducible sample play-by-play data")
    gen.add_argument("--games", type=int, default=240)
    mine = sub.add_parser("mine", help="discover frequent action itemsets")
    mine.add_argument("--min-support", type=float, default=0.08)
    train = sub.add_parser("train", help="fit the local next-action baseline")
    train.add_argument("--context", type=int, default=3)
    train.add_argument("--transformer", action="store_true",
                       help="also continue-pretrain distilgpt2 (requires optional dependencies)")
    train.add_argument("--model-name", default="distilgpt2")
    ev = sub.add_parser("evaluate", help="measure held-out next-action accuracy")
    ev.add_argument("--top-k", type=int, default=3)
    ev.add_argument("--transformer", action="store_true", help="evaluate data/transformer checkpoint")
    pred = sub.add_parser("predict", help="predict likely next actions")
    pred.add_argument("events", nargs="+", help="partial event sequence")
    pred.add_argument("--top-k", type=int, default=5)
    pred.add_argument("--transformer", action="store_true", help="use data/transformer checkpoint")
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Game Action Predictor is ready.")
    elif args.command == "prepare":
        path = generate_games(DATA / "play_by_play.csv", games=args.games)
        print(f"Generated sample data: {path}")
        print(f"Possessions: {len(load_sequences(path))}")
    else:
        data_file = DATA / "play_by_play.csv"
        if not data_file.exists():
            generate_games(data_file)
        sequences = load_sequences(data_file)
        train_data, test_data = split_sequences(sequences)
        if args.command == "mine":
            itemsets = mine_frequent_itemsets(train_data, args.min_support)
            save_itemsets(itemsets, DATA / "frequent_itemsets.json")
            print(f"Found {len(itemsets)} frequent itemsets.")
            for row in itemsets[:20]:
                print(f"{', '.join(row['itemset']):55} support={row['support']:.3f}")
        elif args.command == "train":
            model = NGramActionModel(args.context).fit(train_data)
            model.save(MODEL_PATH)
            print(f"Saved next-action model to {MODEL_PATH}")
            if args.transformer:
                continue_pretrain_transformer(train_data, str(DATA / "transformer"), args.model_name)
                print("Saved continued-pretraining checkpoint to data/transformer")
        elif args.command == "evaluate":
            if not MODEL_PATH.exists():
                raise SystemExit("Train first: python -m app.cli train")
            model = (TransformerActionModel(str(DATA / "transformer"),
                        sorted({a for seq in sequences for a in seq}))
                     if args.transformer else NGramActionModel.load(MODEL_PATH))
            print(json.dumps(evaluate(model, test_data, args.top_k), indent=2))
        elif args.command == "predict":
            if not MODEL_PATH.exists():
                raise SystemExit("Train first: python -m app.cli train")
            print("Most likely next actions:")
            model = (TransformerActionModel(str(DATA / "transformer"),
                        sorted({a for seq in sequences for a in seq}))
                     if args.transformer else NGramActionModel.load(MODEL_PATH))
            for item in model.predict(args.events, args.top_k):
                print(f"  {item['action']:<25} {item['probability']:.1%}")

if __name__ == "__main__":
    main()
