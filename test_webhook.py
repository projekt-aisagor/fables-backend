import requests
import json
from datetime import datetime

def simulate_webhook():
    webhook_url = "http://localhost:9000/webhook/story"
    
    # Simulate the webhook payload
    payload = {
        "type": "INSERT",
        "table": "stories",
        "schema_name": "public",  
        "record": {
            "id": 1,
            "title": "The Adventure Begins",
            "storyline_prompt": "En dag var tårtan borta",
            "minutes_long": 2,
            "world_id": 4,
            "user_id": '73d289d6-f0dd-4035-b12b-e5baf31c1e0c',
            "created_at": datetime.now().isoformat(),
            "story_script": None
        },
        "old_record": None
    }
    
    # Send POST request
    try:
        print("Sending webhook payload:", json.dumps(payload, indent=2))
        response = requests.post(
            webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Print response
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    simulate_webhook()
