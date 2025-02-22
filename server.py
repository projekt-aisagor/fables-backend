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
from concurrent.futures import ThreadPoolExecutor
from pydub import AudioSegment

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
model = "mistral-large-latest"

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

def get_language_name(language_code: str) -> str:
    """Convert ISO language code to full language name."""
    language_map = {
        "ar": "Arabic",
        "bg": "Bulgarian",
        "zh": "Chinese",
        "hr": "Croatian",
        "cs": "Czech",
        "da": "Danish",
        "nl": "Dutch",
        "en": "English",
        "tl": "Filipino",
        "fi": "Finnish",
        "fr": "French",
        "de": "German",
        "el": "Greek",
        "hi": "Hindi",
        "hu": "Hungarian",
        "id": "Indonesian",
        "it": "Italian",
        "ja": "Japanese",
        "ko": "Korean",
        "ms": "Malay",
        "no": "Norwegian",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "ru": "Russian",
        "sk": "Slovak",
        "es": "Spanish",
        "sv": "Swedish",
        "ta": "Tamil",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "vi": "Vietnamese"
    }
    return language_map.get(language_code.lower(), "English")  # Default to English if code not found

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
        
        # Get voice information
        voice_lib = get_voices(supabase)
        voice_info = "Available voices:\n"
        for voice_id, description in voice_lib.items():
            voice_info += f"- {description} (voice_id: {voice_id})\n"
        
        # Generate story script using Mistral
        language_name = get_language_name(language)
        prompt = f"""You are a professional storyteller crafting an engaging audio story in {language_name}. Create a story that:
1. Follows this storyline prompt: {story_record.storyline_prompt}
2. Is approximately {story_record.minutes_long} minutes long when read aloud at a natural pace
3. Takes place in the world: {world.data['name']} - {world.data['description']}
4. Features these characters:
{character_descriptions}
5. Aligns with these base values: {base_values_str}

Your output must be a JSON object with the following structure:
{{
    "title": "Story Title",
    "sections": [
        {{
            "type": "text_to_speech",
            "text": "The actual text to speak",
            "voice_id": "<voice_id from available voices>",
            "model_id": "eleven_multilingual_v2"
        }},
        {{
            "type": "text_to_sound_effects",
            "text": "sound of birds chirping in the forest"
        }},
        {{
            "type": "pause",
            "seconds": 1
        }}
    ]
}}

{voice_info}

Guidelines for sections:
1. Split the story into logical segments at each speaker change or paragraph
2. Remove speaker attributions (e.g., "LUNA:") from the text
3. Convert sound effects (in parentheses) into text_to_sound_effects segments
4. Add 1-second pauses between major segments
5. Use appropriate voices for each character consistently
6. Write all text in {language_name}, except sound effect descriptions which should be in English
7. The story should flow naturally when read aloud and engage listeners through vivid descriptions and compelling dialogue
8. Avoid complex sentence structures or visual-only references that don't translate well to audio

Write the complete story in the specified JSON format:"""
        
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
        
        # Log the raw response for debugging
        raw_content = chat_response.choices[0].message.content
        logger.info("Raw response from Mistral:")
        logger.info(raw_content)
        
        try:
            # Try to clean the response if it contains markdown code blocks
            content = raw_content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]  # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            logger.info("Cleaned content for JSON parsing:")
            logger.info(content)
            
            # Try to parse the JSON
            story_data = json.loads(content)
            
            # Validate the structure
            if not isinstance(story_data, dict):
                raise ValueError("Response is not a JSON object")
            if "title" not in story_data:
                raise ValueError("Missing 'title' in response")
            if "sections" not in story_data:
                raise ValueError("Missing 'sections' in response")
            if not isinstance(story_data["sections"], list):
                raise ValueError("'sections' is not an array")
                
            # Validate each section
            for i, section in enumerate(story_data["sections"]):
                if "type" not in section:
                    raise ValueError(f"Missing 'type' in section {i}")
                if section["type"] == "text_to_speech":
                    required_keys = {"text", "voice_id", "model_id"}
                    missing_keys = required_keys - set(section.keys())
                    if missing_keys:
                        raise ValueError(f"Missing required keys {missing_keys} in text_to_speech section {i}")
                elif section["type"] == "text_to_sound_effects":
                    if "text" not in section:
                        raise ValueError(f"Missing 'text' in sound_effects section {i}")
                elif section["type"] == "pause":
                    if "seconds" not in section:
                        raise ValueError(f"Missing 'seconds' in pause section {i}")
                else:
                    raise ValueError(f"Unknown section type: {section['type']} in section {i}")
            
            logger.info(f"Successfully parsed story data with {len(story_data['sections'])} sections")
            
        except json.JSONDecodeError as e:
            logger.error("JSON parsing error:")
            logger.error(f"Error message: {str(e)}")
            logger.error(f"Error position: char {e.pos}")
            logger.error(f"Line number: {e.lineno}")
            logger.error(f"Column number: {e.colno}")
            logger.error("Content around error:")
            if e.pos > 0:
                start = max(0, e.pos - 50)
                end = min(len(content), e.pos + 50)
                logger.error(f"...{content[start:e.pos]}>>>HERE>>>{content[e.pos:end]}...")
            raise
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while parsing response: {str(e)}")
            raise
            
        logger.info("Story generated successfully")
        
        # Update the story script in Supabase
        try:
            supabase.table("stories")\
                .update({
                    "story_script": json.dumps(story_data),
                    "status": "generating_audio"
                })\
                .eq("id", story_record.id)\
                .execute()
            logger.info(f"Story {story_record.id} updated with generated script")
        except Exception as e:
            logger.error(f"Error updating story in database: {e}")
            return

        # Start audio generation with the sections
        try:
            # Initialize ElevenLabs client
            client = ElevenLabs(
                api_key=os.getenv("ELEVENLABS_API_KEY")
            )
            
            # Generate a random filename for the final MP3
            output_filename = f"{uuid.uuid4()}.mp3"
            output_path = f"/tmp/{output_filename}"
            logger.info(f"Starting parallel audio rendering")
            
            def process_segment(segment):
                """Process a single audio segment. Validates voice IDs and generates audio for text_to_speech segments."""
                # Validate voice ID if it's a text_to_speech segment
                if segment["type"] == "text_to_speech":
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
                            logger.error(f"Invalid voice_id: {voice_id}. Available voices: {list(voice_lib.keys())}")
                            return None

                    # Create a temporary file for this segment
                    temp_filename = f"/tmp/{uuid.uuid4()}.mp3"
                    with open(temp_filename, "wb") as f:
                        result = client.text_to_speech.convert(
                            text=segment["text"],
                            voice_id=segment["voice_id"],
                            model_id=segment["model_id"],
                            output_format="mp3_44100_128",
                        )
                        for chunk in result:
                            f.write(chunk)
                    return temp_filename
                return None
            
            # Process segments in parallel
            temp_files = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                temp_files = list(filter(None, executor.map(process_segment, story_data["sections"])))
            
            # Combine audio files
            if temp_files:
                combined = AudioSegment.from_mp3(temp_files[0])
                for temp_file in temp_files[1:]:
                    audio = AudioSegment.from_mp3(temp_file)
                    combined += audio
                
                # Export the combined audio
                combined.export(output_path, format="mp3")
                
                # Clean up temporary files
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.error(f"Error removing temporary file {temp_file}: {e}")
                
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
        port=9001,
        reload=True,  # Enable hot reloading
        reload_dirs=["."]  # Watch current directory for changes
    )