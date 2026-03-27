from .model import from_pretrained
from .inference import prepare, inference_each


maia2_model = from_pretrained(type="rapid", device="cpu")

prepared = prepare()

fen = "r2qkbnr/ppp2ppp/2npb3/4p3/2B1P3/5N1P/PPPP1PP1/RNBQK2R w KQkq - 0 1"
#1100 - 1900
elo_self = 1400 # poziom osoby wykonujcej ruch
elo_oppo = 1400 # poziom przeciwnika

move_probs, win_prob = inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
print(f"Predicted: {move_probs}, Win Prob: {win_prob}")
#print(f"Correct: {max(move_probs, key=move_probs.get) == move}")