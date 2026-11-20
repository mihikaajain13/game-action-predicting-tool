from app.data import generate_games, load_sequences
from app.mining import mine_frequent_itemsets
from app.model import NGramActionModel, evaluate

def test_end_to_end(tmp_path):
    path = generate_games(tmp_path / "plays.csv", games=10)
    sequences = load_sequences(path)
    assert len(sequences) == 200
    itemsets = mine_frequent_itemsets(sequences[:160], min_support=0.05)
    assert itemsets
    model = NGramActionModel().fit(sequences[:160])
    assert model.predict(sequences[160][:2])
    assert evaluate(model, sequences[160])["examples"] > 0
