import sys
from supabase import create_client, ClientOptions

url = "http://127.0.0.1:54321"
key = "YOUR_LOCAL_SUPABASE_SERVICE_ROLE_KEY"

print(f"Connecting to {url} with key {key[:10]}...")

try:
    client = create_client(
        url, 
        key,
        options=ClientOptions(
            auto_refresh_token=False,
            persist_session=False
        )
    )
    # Try a simple query
    print("Attempting to list tables (via firm table select)...")
    try:
        response = client.table("firm").select("count", count="exact").execute()
        print(f"Success! Response: {response}")
    except Exception as e:
        print(f"Query failed: {e}")
        # Even if table doesn't exist, we might get a specific error proving auth worked
        if "relation" in str(e) and "does not exist" in str(e):
             print("Auth likely worked, but table missing.")
        
except Exception as e:
    print(f"Connection failed: {e}")
