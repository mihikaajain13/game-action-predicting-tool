from app.data import generate_games, load_sequences, split_sequences
from app.mining import mine_frequent_itemsets
from app.model import NGramActionModel, evaluate

def test_data_generation_and_basket_format(tmp_path):
    path = generate_games(tmp_path / "events.csv", games=4, seed=7)
    sequences = load_sequences(path)
    assert len(sequences) == 80
    assert all(isinstance(seq, list) and seq for seq in sequences)
    assert all(isinstance(action, str) for seq in sequences for action in seq)

def test_apriori_finds_known_frequent_pair():
    baskets = [["A", "B"], ["A", "B"], ["A", "C"], ["B", "C"]]
    found = mine_frequent_itemsets(baskets, min_support=0.5)
    pair = next(row for row in found if row["itemset"] == ["A", "B"])
    assert pair["support_count"] == 2
    assert pair["support"] == 0.5

def test_model_learns_and_roundtrips(tmp_path):
    sequences = [["rebound", "pass", "layup"], ["rebound", "pass", "layup"],
                 ["steal", "drive", "layup"]]
    model = NGramActionModel().fit(sequences)
    assert model.predict(["rebound", "pass"])[0]["action"] == "layup"
    path = tmp_path / "model.json"
    model.save(path)
    loaded = NGramActionModel.load(path)
    assert loaded.predict(["rebound", "pass"])[0]["action"] == "layup"

def test_evaluation_is_bounded():
    sequences = [["a", "b", "c"], ["a", "b", "c"]]
    train, test = split_sequences(sequences, 0.5)
    result = evaluate(NGramActionModel().fit(train), test)
    assert result["examples"] == 2
    assert 0 <= result["hits_at_k"] <= 1
