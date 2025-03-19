from fastapi import FastAPI
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from pydantic import BaseModel
from utils.essayAgent import EssayWriter
from utils.Agentprompts import *
from utils.guardrail import guard
from dotenv import load_dotenv

load_dotenv()

class user(BaseModel):
    topic: str
    max_revisions: int

model = ChatOpenAI(model="gpt-4o-mini")

tool = TavilySearchResults(
    max_results=2,
    search_depth="basic",
    include_answer=True,
    include_raw_content=True,
    include_images=True
)

agent = EssayWriter(tool, model)
app = FastAPI()

@app.post("/generateEssay")
def essay_write(data: user):

    try:
        query = guard.validate(data.topic)

        res = agent.graph.invoke({
            "task": query,
            "plan_prompt": PLAN_PROMPT,
            "research_prompt": RESEARCH_PLAN_PROMPT,
            "write_prompt": WRITER_PROMPT,
            "reflection_prompt": REFLECTION_PROMPT,
            "critique_research_prompt": RESEARCH_CRITIQUE_PROMPT,
            "max_revisions": data.max_revisions
        })

        return {"final_draft": res['essay_draft']}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

