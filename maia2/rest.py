from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import from_pretrained
from .inference import prepare, inference_each

# how to run:
# python -m maia2.rest 

app = FastAPI(title="chess-evaluation")

# Load model and prepare shared data once
maia2_model = from_pretrained(type="rapid", device="cpu")
prepared = prepare()


class EvalRequest(BaseModel):
    fen: str
    playerElo: int # elo of user that is on move
    opponentElo: int # this is used to calculate win probability


class EvalResponse(BaseModel):
    bestMoves: dict
    whiteWinProbability: float


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(req: EvalRequest):
    """Evaluate a single FEN position.

    Request JSON: {"fen": "...", "playerElo": 1400, "opponentElo": 1400}
    """
    try:
        move_probs, win_prob = inference_each(maia2_model, prepared, req.fen, req.playerElo, req.opponentElo)
    except Exception as e:
        # Return a 400 with the exception message for client debugging
        raise HTTPException(status_code=400, detail=str(e))

    return {"bestMoves": move_probs, "whiteWinProbability": win_prob}


if __name__ == "__main__":
    # Run with: python -m maia2.service
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8085)
