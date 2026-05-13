import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

class DMRequest(BaseModel):
    service: str
    audience: str
    platform: str
    tone: str
    goal: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/generate")
def generate_dms(req: DMRequest):
    if not req.service.strip() or not req.audience.strip():
        raise HTTPException(status_code=400, detail="Service and audience are required.")

    prompt = f"""You are an expert copywriter specializing in cold outreach messages that actually get replies.

Generate cold DMs for the following context:
- Service/Offer: {req.service}
- Target Audience: {req.audience}
- Platform: {req.platform}
- Tone: {req.tone}
- Goal: {req.goal}

Decide how many DMs to generate (between 5 and 10) based on the richness of the context provided.

Rules:
- Each DM must feel human, not robotic
- No emojis unless platform is Instagram
- Keep each DM under 150 words
- Each DM should have a different angle/hook
- Never start two DMs with the same word
- Make them platform-appropriate

Respond ONLY with a valid JSON array. No extra text, no markdown, no backticks.

Format:
[
  {{
    "label": "Hook type (e.g. Pain Point, Curiosity, Social Proof, Direct Ask, Compliment, Results-First, Question Hook)",
    "message": "The full DM text here"
  }}
]"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        raw = response.text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        dms = json.loads(raw)
        return {"dms": dms, "count": len(dms)}

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse AI response. Try again.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))