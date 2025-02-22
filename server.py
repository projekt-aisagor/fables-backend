import os
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv
from supabase import create_client, Client
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging
import uuid
from elevenlabs.client import ElevenLabs
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

@app.middleware("http")
async def log_request_body(request: Request, call_next):
    if request.url.path == "/webhook/story":
        try:
            body = await request.body()
            logger.info(f"Raw incoming webhook request body: {body.decode()}")
        except Exception as e:
            logger.error(f"Error reading request body: {e}")
    response = await call_next(request)
    return response

# Initialize Mistral client
mistral_api_key = os.environ["MISTRAL_API_KEY"]
mistral_client = MistralClient(api_key=mistral_api_key)
model = "ministral-3b-latest" #"mistral-large-latest"

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class StoryRecord(BaseModel):
    id: int
    title: Optional[str] = None
    storyline_prompt: str
    minutes_long: int
    world_id: int
    user_id: uuid.UUID
    created_at: datetime
    story_script: Optional[str] = None
    status: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryRecord':
        logger.info(f"Creating StoryRecord from data: {data}")
        # Convert string to UUID for user_id if it's a string
        if isinstance(data.get('user_id'), str):
            data['user_id'] = uuid.UUID(data['user_id'])
        
        # Use storyline_prompt as title if title is not provided
        if 'title' not in data:
            data['title'] = data.get('storyline_prompt')
            
        return cls(**data)

class WebhookPayload(BaseModel):
    type: str
    table: str
    schema: str  # Changed from db_schema to match actual payload
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

def get_voices(supabase_client: Client) -> Dict[str, str]:
    """Fetch available voices from Supabase."""
    response = supabase_client.table('voices').select('id, description').execute()
    return {voice['id']: voice['description'] for voice in response.data}

def parse_script_for_tts(story_script: str, characters: List[Dict]) -> List[Dict]:
    """Ask Mistral to parse the script and generate text-to-speech and sound effect segments"""
    
    voice_lib = get_voices(supabase)
    
    def validate_voice_id(segment: Dict) -> Dict:
        """Validate and potentially fix voice_id in a segment"""
        if segment["type"] != "text_to_speech":
            return segment
            
        voice_id = segment["voice_id"]
        # If the voice_id is not in our library, try to find it by name
        if voice_id not in voice_lib:
            # Create a mapping of names to IDs
            name_to_id = {
                name.split(',')[0].strip(): id  # Take first part before comma as name
                for id, desc in voice_lib.items()
                for name in [desc.split(',')[0].strip()]  # Extract name before first comma
            }
            # Try to find the voice ID by name
            if voice_id in name_to_id:
                segment["voice_id"] = name_to_id[voice_id]
            else:
                raise ValueError(f"Invalid voice_id: {voice_id}. Available voices: {list(voice_lib.keys())}")
        
        return segment

    # Generate voice information text from Supabase data
    voice_info = "Available voices:\n"
    for voice_id, description in voice_lib.items():
        voice_info += f"- {description} (voice_id: {voice_id})\n"
    
    # Create the prompt for Mistral
    prompt = f"""You are a script parser that converts story scripts into audio segments.

Task: Parse the following script and output ONLY a JSON array of audio segments.

{voice_info}

Required format for each segment:
For speech:
{{
    "type": "text_to_speech",
    "text": "The actual text to speak",
    "voice_id": "<use appropriate voice_id from above>",
    "model_id": "eleven_multilingual_v2"
}}

For sound effects (convert descriptions to English):
{{
    "type": "text_to_sound_effects",
    "text": "sound of birds chirping in the forest"
}}

For pauses:
{{
    "type": "pause",
    "seconds": 1
}}

Rules:
1. Split the script into logical segments at each speaker change or paragraph
2. Remove speaker attributions (e.g., "LUNA:") from the text
3. Convert sound effects (in parentheses) into English text_to_sound_effects segments
4. Add 1-second pauses between major segments
5. Ensure the output is valid JSON
6. Return ONLY the JSON array, no explanations or other text

Example output:
[
    {{
        "type": "text_to_speech",
        "text": "Once upon a time in a magical forest...",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_multilingual_v2"
    }},
    {{
        "type": "text_to_sound_effects",
        "text": "sound of gentle wind rustling through leaves"
    }},
    {{
        "type": "pause",
        "seconds": 1
    }},
    {{
        "type": "text_to_speech",
        "text": "Who goes there? Show yourself!",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_multilingual_v2"
    }}
]

Here is the script to parse:

{story_script}

Remember: Output ONLY the JSON array, nothing else."""

    try:
        # Get Mistral's response
        messages = [
            ChatMessage(
                role="user",
                content=prompt
            )
        ]
        
        response = mistral_client.chat(
            messages=messages,
            model=model
        )
        
        # Get the content and debug log it
        content = response.choices[0].message.content
        logger.info(f"Raw Mistral response:\n{content}")
        
        # Try to find JSON array in the response
        content = content.strip()
        if not content.startswith('['):
            start_idx = content.find('[')
            if start_idx != -1:
                content = content[start_idx:]
            else:
                raise ValueError("No JSON array found in response")
        
        if not content.endswith(']'):
            end_idx = content.rfind(']')
            if end_idx != -1:
                content = content[:end_idx+1]
            else:
                raise ValueError("No JSON array end found in response")
        
        # Parse the JSON
        segments = json.loads(content)
        
        # Validate and potentially fix voice IDs
        segments = [validate_voice_id(segment) for segment in segments]
        
        # Validate the structure
        if not isinstance(segments, list):
            raise ValueError("Response is not a JSON array")
        
        for segment in segments:
            if "type" not in segment:
                raise ValueError(f"Missing 'type' in segment: {segment}")
                
            if segment["type"] == "text_to_speech":
                required_keys = {"text", "voice_id", "model_id"}
                if not all(key in segment for key in required_keys):
                    raise ValueError(f"Missing required keys in text_to_speech segment: {segment}")
            elif segment["type"] == "text_to_sound_effects":
                if "text" not in segment:
                    raise ValueError(f"Missing 'text' in sound_effects segment: {segment}")
            elif segment["type"] == "pause":
                if "seconds" not in segment:
                    raise ValueError(f"Missing 'seconds' in pause segment: {segment}")
            else:
                raise ValueError(f"Unknown segment type: {segment['type']}")
        
        logger.info(f"Successfully parsed {len(segments)} audio segments")
        return segments
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing segments: {str(e)}")

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
        
        # Update the story in Supabase with the script
        try:
            supabase.table("stories")\
                .update({"story_script": story_script, "status": "generating_audio"})\
                .eq("id", story_record.id)\
                .execute()
            logger.info(f"Story {story_record.id} updated with generated script")
        except Exception as e:
            logger.error(f"Error updating story in database: {e}")
            return

        # Parse the script into segments for audio generation
        try:
            segments = parse_script_for_tts(story_script, characters.data)
            
            # Initialize ElevenLabs client
            client = ElevenLabs(
                api_key=os.getenv("ELEVENLABS_API_KEY")
            )
            
            # Generate a random filename for the MP3
            output_filename = f"{uuid.uuid4()}.mp3"
            output_path = f"/tmp/{output_filename}"
            logger.info(f"Starting audio rendering to {output_path}")
            
            # Render all segments
            with open(output_path, "wb") as f:
                for segment in segments:
                    logger.info(f"Processing segment type: {segment['type']}")
                    
                    if segment["type"] == "text_to_speech":
                        result = client.text_to_speech.convert(
                            text=segment["text"],
                            voice_id=segment["voice_id"],
                            model_id=segment["model_id"],
                            output_format="mp3_44100_128"
                        )
                        for chunk in result:
                            f.write(chunk)
                    
                    elif segment["type"] == "pause":
                        # For now, we skip pauses as they need to be handled differently
                        continue
            
            logger.info(f"Successfully rendered story to {output_path}")

            # Upload the file to Supabase Storage
            try:
                with open(output_path, 'rb') as f:
                    supabase.storage.from_("story-audio").upload(
                        path=output_filename,
                        file=f,
                        file_options={"content-type": "audio/mpeg"}
                    )
                logger.info(f"Successfully uploaded audio file to Supabase Storage: {output_filename}")

                # Update the story record with the audio file path and set status to ready
                supabase.table("stories")\
                    .update({
                        "audio_file": output_filename,
                        "status": "done"
                    })\
                    .eq("id", story_record.id)\
                    .execute()
                logger.info(f"Story {story_record.id} marked as ready with audio file {output_filename}")

                # Clean up the temporary file
                os.remove(output_path)
                logger.info(f"Cleaned up temporary file {output_path}")

            except Exception as e:
                logger.error(f"Error uploading to Supabase Storage: {e}")
                # Update status to error
                supabase.table("stories")\
                    .update({"status": "error"})\
                    .eq("id", story_record.id)\
                    .execute()
                
        except Exception as e:
            logger.error(f"Error in audio generation process: {e}")
            # Update status to error
            supabase.table("stories")\
                .update({"status": "error"})\
                .eq("id", story_record.id)\
                .execute()
            
    except Exception as e:
        logger.error(f"Error in story generation process: {e}")
        # Update status to error
        try:
            supabase.table("stories")\
                .update({"status": "error"})\
                .eq("id", story_record.id)\
                .execute()
        except:
            pass

@app.post("/webhook/story", response_model=WebhookResponse)
async def handle_story_webhook(request: Request, payload: WebhookPayload, background_tasks: BackgroundTasks):
    # Log raw request body for debugging
    body = await request.json()
    logger.info(f"Raw webhook request body: {json.dumps(body, indent=2)}")
    logger.info(f"Received webhook payload type: {payload.type}")
    logger.info(f"Received webhook payload table: {payload.table}")
    logger.info(f"Received webhook payload schema: {payload.schema}")
    logger.info(f"Received webhook payload record: {json.dumps(payload.record, indent=2)}")
    
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