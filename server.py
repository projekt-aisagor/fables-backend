import os
from fastapi import FastAPI, Request, HTTPException
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Mistral client
mistral_api_key = os.environ["MISTRAL_API_KEY"]
mistral_client = MistralClient(api_key=mistral_api_key)
model = "mistral-large-latest"

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class StoryRecord(BaseModel):
    id: int
    title: str
    storyline_prompt: str
    minutes_long: int
    world_id: int
    user_id: uuid.UUID
    created_at: datetime
    story_script: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryRecord':
        logger.info(f"Creating StoryRecord from data: {data}")
        # Convert string to UUID for user_id if it's a string
        if isinstance(data.get('user_id'), str):
            data['user_id'] = uuid.UUID(data['user_id'])
        return cls(**data)

class WebhookPayload(BaseModel):
    type: str
    table: str
    schema_name: str  # Renamed from schema to avoid conflict
    record: Dict[str, Any]  # Changed to Dict for initial parsing
    old_record: Optional[Dict[str, Any]] = None  # Changed to Dict for initial parsing

    class Config:
        from_attributes = True

class WebhookResponse(BaseModel):
    status: str
    story_id: Optional[int] = None
    message: Optional[str] = None

    class Config:
        from_attributes = True

@app.post("/webhook/story", response_model=WebhookResponse)
async def handle_story_webhook(payload: WebhookPayload):
    logger.info(f"Received webhook payload: {payload}")
    
    if payload.table != "stories":
        logger.info("Ignoring non-stories table webhook")
        return WebhookResponse(
            status="ignored",
            message="not stories table"
        )
    
    try:
        # Convert record dict to StoryRecord
        logger.info("Converting record to StoryRecord")
        story_record = StoryRecord.from_dict(payload.record)
        logger.info(f"Created StoryRecord: {story_record}")
        
        # Fetch additional data from Supabase
        logger.info("Fetching additional data from Supabase")
        
        # Get world data
        try:
            world = supabase.table("worlds").select("*").eq("id", story_record.world_id).single().execute()
            if not world.data:
                return WebhookResponse(
                    status="error",
                    message=f"World with id {story_record.world_id} not found in database. Please create the world first."
                )
        except Exception as e:
            logger.error(f"Error fetching world data: {e}")
            return WebhookResponse(
                status="error",
                message=f"World with id {story_record.world_id} not found in database. Please create the world first."
            )
        
        # Get character data
        try:
            characters = supabase.table("story_characters")\
                .select("characters(*)").eq("story_id", story_record.id).execute()
            if not characters.data:
                return WebhookResponse(
                    status="error",
                    message=f"No characters found for story {story_record.id}. Please add at least one character to the story."
                )
        except Exception as e:
            logger.error(f"Error fetching character data: {e}")
            return WebhookResponse(
                status="error",
                message=f"Error fetching characters for story {story_record.id}. Please ensure characters are properly set up."
            )
        
        # Get base values for the user
        try:
            base_values = supabase.table("base_values")\
                .select("value").eq("user_id", story_record.user_id).execute()
            base_values_list = [v["value"] for v in base_values.data] if base_values.data else []
            if not base_values_list:
                logger.warning(f"No base values found for user {story_record.user_id}")
        except Exception as e:
            logger.error(f"Error fetching base values: {e}")
            base_values_list = []
        
        # Format character descriptions
        character_descriptions = "\n".join([
            f"- {char['characters']['name']}: {char['characters']['about']}" 
            for char in characters.data
        ])
        
        # Format base values string
        base_values_str = ", ".join(base_values_list)
        
        # Generate story script using Mistral
        prompt = f"""You are a professional storyteller crafting an engaging audio story. Create a story that:
1. Follows this storyline prompt: {story_record.storyline_prompt}
2. Is approximately {story_record.minutes_long} minutes long when read aloud at a natural pace
3. Takes place in the world: {world.data['name']} - {world.data['description']}
4. Features these characters:
{character_descriptions}
5. Aligns with these base values: {base_values_str}

Format the story as a professional audio script with:
- Clear speaker attribution for dialogue
- Natural pacing and pauses indicated by [...] for dramatic effect
- Sound effect suggestions in parentheses (if appropriate)
- Proper pronunciation guides for unusual names in [square brackets]
- Scene transitions marked with subtle audio cues

The story should flow naturally when read aloud and engage listeners through vivid descriptions and compelling dialogue. Avoid complex sentence structures or visual-only references that don't translate well to audio.

Write the complete story script:"""
        
        logger.info(f"Generating story with prompt: {prompt}")
        
        messages = [
            ChatMessage(
                role="user",
                content=prompt
            )
        ]
        
        chat_response = mistral_client.chat(
            model=model,
            messages=messages
        )
        
        generated_story = chat_response.choices[0].message.content
        logger.info("Successfully generated story")
        logger.info(f"Generated story: {generated_story[:100]}...")  # Log first 100 chars
        
        # Update the story in Supabase
        logger.info(f"Updating story in Supabase for ID: {story_record.id}")
        result = supabase.table("stories") \
            .update({"story_script": generated_story}) \
            .eq("id", story_record.id) \
            .execute()
        logger.info("Successfully updated story in Supabase")
            
        return WebhookResponse(
            status="success",
            story_id=story_record.id
        )
        
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}", exc_info=True)
        return WebhookResponse(
            status="error",
            message=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=9000,
        reload=True,  # Enable hot reloading
        reload_dirs=["."]  # Watch current directory for changes
    )