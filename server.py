import os
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
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

async def generate_story(story_record: StoryRecord, supabase: Client, mistral: MistralClient):
    """Background task to generate and save the story"""
    logger.info(f"Starting async story generation for story {story_record.id}")
    
    try:
        # Fetch additional data from Supabase
        logger.info("Fetching additional data from Supabase")
        
        # Get world data
        try:
            world = supabase.table("worlds").select("*").eq("id", story_record.world_id).single().execute()
            if not world.data:
                logger.error(f"World with id {story_record.world_id} not found in database")
                return
        except Exception as e:
            logger.error(f"Error fetching world data: {e}")
            return
        
        # Get character data
        try:
            characters = supabase.table("story_characters")\
                .select("characters(*)").eq("story_id", story_record.id).execute()
            if not characters.data:
                logger.error(f"No characters found for story {story_record.id}")
                return
        except Exception as e:
            logger.error(f"Error fetching character data: {e}")
            return
        
        # Get base values and language for the user
        try:
            base_values = supabase.table("base_values")\
                .select("value, language")\
                .eq("user_id", story_record.user_id)\
                .execute()
            
            # Extract values and language
            base_values_list = []
            language = "en"  # Default to English if not specified
            
            if base_values.data:
                base_values_list = [v["value"] for v in base_values.data if v.get("value")]
                # Use the language from the first row that has it defined
                for value in base_values.data:
                    if value.get("language"):
                        language = value["language"]
                        break
                        
            if not base_values_list:
                logger.warning(f"No base values found for user {story_record.user_id}")
            logger.info(f"Using language: {language}")
            
        except Exception as e:
            logger.error(f"Error fetching base values: {e}")
            base_values_list = []
            language = "en"  # Default to English on error
        
        # Format character descriptions
        character_descriptions = "\n".join([
            f"- {char['characters']['name']}: {char['characters']['about']}" 
            for char in characters.data
        ])
        
        # Format base values string
        base_values_str = ", ".join(base_values_list)
        
        # Generate story script using Mistral
        prompt = f"""You are a professional storyteller crafting an engaging audio story in {language}. Create a story that:
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

IMPORTANT: Write the complete story script in {language}. Make sure all text, including speaker attributions and sound effects, is in {language}.

Write the complete story script:"""
        
        logger.info(f"Generating story with prompt: {prompt}")
        
        messages = [
            ChatMessage(
                role="user",
                content=prompt
            )
        ]
        
        # Generate the story
        chat_response = mistral.chat(
            messages=messages,
            model=model
        )
        
        story_script = chat_response.choices[0].message.content
        logger.info("Story generated successfully")
        
        # Update the story in Supabase
        try:
            supabase.table("stories")\
                .update({"story_script": story_script})\
                .eq("id", story_record.id)\
                .execute()
            logger.info(f"Story {story_record.id} updated with generated script")
        except Exception as e:
            logger.error(f"Error updating story in database: {e}")
            
    except Exception as e:
        logger.error(f"Error in story generation process: {e}")

@app.post("/webhook/story", response_model=WebhookResponse)
async def handle_story_webhook(payload: WebhookPayload, background_tasks: BackgroundTasks):
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
        
        # Queue the story generation as a background task
        background_tasks.add_task(
            generate_story,
            story_record=story_record,
            supabase=supabase,
            mistral=mistral_client
        )
        
        return WebhookResponse(
            status="accepted",
            story_id=story_record.id,
            message="Story generation started"
        )
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
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