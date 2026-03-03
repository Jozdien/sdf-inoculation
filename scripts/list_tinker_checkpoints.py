"""List all available Tinker checkpoints for the authenticated user."""

import os

import tinker
from dotenv import load_dotenv

load_dotenv()  # loads TINKER_API_KEY from .env

if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError(
        "TINKER_API_KEY not found. Add it to your .env file:\n"
        "  TINKER_API_KEY=your-key-here"
    )

service_client = tinker.ServiceClient()
rest_client = service_client.create_rest_client()

# Fetch all checkpoints across all training runs for this user
response = rest_client.list_user_checkpoints(limit=100).result()

print(f"Found {len(response.checkpoints)} checkpoint(s)")
if response.cursor:
    print(f"(Total: {response.cursor.total_count})")
print()

if response.checkpoints:
    # Introspect the actual fields on the Checkpoint model
    first = response.checkpoints[0]
    print(f"Checkpoint fields: {list(first.model_fields.keys())}")
    print(f"Example: {first}")
    print()

    for cp in response.checkpoints:
        print(f"  {cp}")
