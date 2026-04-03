import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from env import InvoiceEnv

app = FastAPI()
env = InvoiceEnv()

@app.get("/")
def home():
    return {
        "name": "Invoice Processing Environment",
        "version": "1.0.0",
        "status": "running",
        "endpoints": ["/reset", "/step", "/state"]
    }

@app.get("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    from models import Action
    act = Action(**action)
    obs, reward, done, info = env.step(act)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()