import os
from dotenv import load_dotenv
from supabase import create_client, Client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from elevenlabs.client import ElevenLabs
from typing import List, Dict
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Initialize Mistral client
mistral = MistralClient(
    api_key=os.getenv("MISTRAL_API_KEY")
)

def fetch_story_data(story_id: int) -> Dict:
    """Fetch story and related data from Supabase"""
    try:
        # Get story data
        story = supabase.table("stories").select("*").eq("id", story_id).single().execute()
        if not story.data:
            raise ValueError(f"Story with id {story_id} not found")
        
        # Get characters for the story
        characters = supabase.table("story_characters")\
            .select("characters(*)")\
            .eq("story_id", story_id)\
            .execute()
        
        return {
            "story": story.data,
            "characters": [char["characters"] for char in characters.data] if characters.data else []
        }
    except Exception as e:
        logger.error(f"Error fetching story data: {e}")
        raise

def parse_script_for_tts(story_script: str, characters: List[Dict]) -> List[Dict]:
    """Ask Mistral to parse the script and generate text-to-speech and sound effect segments"""
    
    # Create a character name to ID mapping
    char_map = {char["name"]: char["id"] for char in characters}
    
    # Create the prompt for Mistral
    prompt = f"""You are a script parser that converts story scripts into audio segments.

Task: Parse the following script and output ONLY a JSON array of audio segments.

Required format for each segment:
For speech:
{{
    "type": "text_to_speech",
    "text": "The actual text to speak",
    "voice_id": "JBFqnCBsd6RMkjVDRZzb",
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
        
        response = mistral.chat(
            messages=messages,
            model="mistral-large-latest"
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
        logger.error(f"JSON parsing error: {e}")
        logger.error(f"Problematic content: {content}")
        raise
    except Exception as e:
        logger.error(f"Error parsing script: {e}")
        raise

def render_story(story_id: int):
    """Main function to render a story"""
    try:
        # Fetch story data
        story_data = fetch_story_data(story_id)
        story = story_data["story"]
        characters = story_data["characters"]
        
        # Parse the script into segments
        segments = parse_script_for_tts(story["story_script"], characters)
        
        # Initialize ElevenLabs client
        client = ElevenLabs(
            api_key=os.getenv("ELEVENLABS_API_KEY")
        )
        
        output_path = "output.mp3"
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
                        output_format="mp3_44100_128", 
                    )
                    for chunk in result:
                        f.write(chunk)
                        
                elif segment["type"] == "text_to_sound_effects":
                    # TODO: figure out why sound effects are not working?
                    if False:
                        result = client.text_to_sound_effects.convert(
                            text=segment["text"],
                        )
                        for chunk in result:
                             f.write(chunk)        
                elif segment["type"] == "pause":
                    # For now, we skip pauses as they need to be handled differently
                    # TODO: Implement proper pause handling
                    continue
                    
        logger.info(f"Successfully rendered story to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error rendering story: {e}")
        raise

if __name__ == "__main__":
    render_story(1)
