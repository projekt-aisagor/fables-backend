import os
from dotenv import load_dotenv
from supabase import create_client, Client
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
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
    """Ask Mistral to parse the script and generate text-to-speech tool calls"""
    
    # Create a character name to ID mapping
    char_map = {char["name"]: char["id"] for char in characters}
    
    # Create the prompt for Mistral
    prompt = f"""You are a script parser that converts story scripts into text-to-speech segments.

Task: Parse the following script and output ONLY a JSON array of text-to-speech tool calls.

Required format for each segment:
{{
    "text": "The actual text to speak",
    "voice_id": "JBFqnCBsd6RMkjVDRZzb",
    "model_id": "eleven_multilingual_v2",
    "output_format": "mp3_44100_128"
}}

Rules:
1. Split the script into logical segments at each speaker change or paragraph
2. Remove speaker attributions (e.g., "LUNA:") from the text
3. Keep pauses [...] and sound effects (in parentheses) in the text
4. Ensure the output is valid JSON
5. Return ONLY the JSON array, no explanations or other text

Example output:
[
    {{
        "text": "Once upon a time in a magical forest...",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_128"
    }},
    {{
        "text": "Who goes there? [...] Show yourself!",
        "voice_id": "JBFqnCBsd6RMkjVDRZzb",
        "model_id": "eleven_multilingual_v2",
        "output_format": "mp3_44100_128"
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
            # Try to find the start of the JSON array
            start_idx = content.find('[')
            if start_idx != -1:
                content = content[start_idx:]
            else:
                raise ValueError("No JSON array found in response")
        
        if not content.endswith(']'):
            # Try to find the end of the JSON array
            end_idx = content.rfind(']')
            if end_idx != -1:
                content = content[:end_idx+1]
            else:
                raise ValueError("No JSON array end found in response")
        
        # Parse the JSON
        tool_calls = json.loads(content)
        
        # Validate the structure
        if not isinstance(tool_calls, list):
            raise ValueError("Response is not a JSON array")
        
        for call in tool_calls:
            required_keys = {"text", "voice_id", "model_id", "output_format"}
            if not all(key in call for key in required_keys):
                raise ValueError(f"Missing required keys in tool call: {call}")
        
        logger.info(f"Successfully parsed {len(tool_calls)} text-to-speech segments")
        return tool_calls
        
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
        logger.info(f"Fetching data for story {story_id}")
        story_data = fetch_story_data(story_id)
        
        if not story_data["story"].get("story_script"):
            logger.error("Story script is empty")
            return
        
        # Parse script and generate TTS tool calls
        logger.info("Parsing script for text-to-speech")
        tts_segments = parse_script_for_tts(
            story_data["story"]["story_script"],
            story_data["characters"]
        )
        
        # Print the tool calls (for now)
        logger.info("Generated text-to-speech segments:")
        for i, segment in enumerate(tts_segments, 1):
            print(f"\nSegment {i}:")
            print(json.dumps(segment, indent=2))
            
    except Exception as e:
        logger.error(f"Error rendering story: {e}")
        raise

if __name__ == "__main__":
    render_story(1)
