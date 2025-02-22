import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

def setup_test_data():
    # Create test world
    world_data = {
        "id": 1,
        "name": "The Enchanted Forest",
        "description": "A mystical forest where ancient magic flows through every tree and creature",
        "user_id": 1
    }
    try:
        world = supabase.table("worlds").upsert(world_data).execute()
        print("Created test world:", world.data)
    except Exception as e:
        print(f"Error creating world: {e}")

    # Create test character
    character_data = {
        "id": 1,
        "name": "Luna",
        "about": "A wise forest spirit who guides lost travelers",
        "user_id": 1
    }
    try:
        character = supabase.table("characters").upsert(character_data).execute()
        print("Created test character:", character.data)
    except Exception as e:
        print(f"Error creating character: {e}")

    # Create story-character relationship
    story_character_data = {
        "story_id": 1,
        "character_id": 1
    }
    try:
        story_character = supabase.table("story_characters").upsert(story_character_data).execute()
        print("Created story-character relationship:", story_character.data)
    except Exception as e:
        print(f"Error creating story-character relationship: {e}")

    # Create test base values
    base_values_data = [
        {"user_id": 1, "value": "friendship"},
        {"user_id": 1, "value": "discovery"},
        {"user_id": 1, "value": "harmony with nature"}
    ]
    try:
        base_values = supabase.table("base_values").upsert(base_values_data).execute()
        print("Created test base values:", base_values.data)
    except Exception as e:
        print(f"Error creating base values: {e}")

if __name__ == "__main__":
    setup_test_data()
