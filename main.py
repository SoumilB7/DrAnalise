from typing import Union  # uvicorn main:app --reload
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from scipy.spatial.distance import cosine # pip install scripy
from sentence_transformers import SentenceTransformer, util
from pymongo import MongoClient
from pydantic import BaseModel


# --------------------------Mongo DB conversation ------------------------

conn = MongoClient()

# Yet to be made
def dataset_initialize():
    append = False
    # using Mongo DB
    # I will likely get a set of ideas that I have fetch from the dataset
    # Return (That list)

def dataset_append(idea):
    append = True
    # using Mongo DB
    # here I will append ideas to that Mongo DB dataset
    # return (Conformation status)

# -------------------------------------------------------------------------


# ------------------------------------- Model ------------------------------------------     

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
class SimilarityChecker:
    def __init__(self, threshold=0.85, batch_size=32):
        self.threshold = threshold
        self.batch_size = batch_size
        self.use_model = model
        self.data = []
    
    def data_initialize(self):
        self.data = dataset_initialize()  # initializing to the og one 

    
    def check_similarity(self, new_idea):
        new_vector = self.use_model.encode([new_idea], convert_to_tensor=True)
        num_batches = (len(self.data) + self.batch_size - 1) // self.batch_size
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = (i + 1) * self.batch_size
            old_batch = self.data[start_idx:end_idx]
            old_vectors = self.use_model.encode(old_batch, convert_to_tensor=True)
            similarity_scores = util.pytorch_cos_sim(new_vector, old_vectors)
            max_similarity_score = similarity_scores.max()
            if max_similarity_score > self.threshold:
                return True, max_similarity_score.item()
        return False, None



def idea_chk(ideam):    # this will check the dataset(txt file) for similar ideas 
    similarity_checker = SimilarityChecker()
    similarity_checker.data_initialize()
    is_similar, similarity_percent = similarity_checker.check_similarity(ideam)
    if not is_similar:
        return True  # idea is new
    else:
        return False # idea is similar

# Download upto
# ---------------------------------------------------------------------------------------


# ------------------------------------- Shaping Request ---------------------------------

class Request(BaseModel):
    perform: str | None = None # append
    idea_set: list  # all the set of ideas that have been stacked along the way

# ---------------------------------------------------------------------------------------


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")



@app.get("/")   # front page yeahh
async def read_root():
    return {"Post at": "/append/"}


@app.get("/items/{q}")  # these are for specific branches of the backend you want to go 
async def read_item( q: Union[str, None] = None):
    return { "q": q}

# --------------------------------- Appending area ---------------------------------------
@app.post("/append/")
async def get_req(info: Request):
    ideas = info.idea_set
    for idea in ideas:
        if idea_chk(idea):
            status = dataset_append(idea)
    return status # 200 for healthy conversation 