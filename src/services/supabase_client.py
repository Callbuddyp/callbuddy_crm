from typing import Dict, List, Optional, Tuple, Any
import os
import sys
from supabase import create_client, Client, ClientOptions

import logging

logger = logging.getLogger(__name__)

class SupabaseService:
    def __init__(self, url: str, key: str):
        if not url or not key:
            raise ValueError("Supabase URL and Key must be provided")
        
        # Disable auto-refresh as this is a script
        self.client: Client = create_client(
            url, 
            key,
            options=ClientOptions(
                auto_refresh_token=False,
                persist_session=False
            )
        )
    
    def get_or_create_firm(self, name: str) -> str:
        """Get existing firm by name or create a new one. Returns firm_id.
        
        Args:
            name: Firm name to look up or create with
        """
        if not name:
            raise ValueError("Firm name must be provided")
        
        # Try to find existing by name
        try:
            response = self.client.table("firm").select("*").eq("firm_name", name).single().execute()
            if response.data:
                logger.info(f"ℹ️ Firm already exists: {name}")
                return response.data["id"]
        except Exception as e:
            # Code PGRST116 means no rows found, which is expected
            if "PGRST116" not in str(e):
                logger.error(f"Error checking firm existence: {e}")
                raise

        # Create new
        try:
            response = self.client.table("firm").insert({"firm_name": name}).execute()
            if response.data and len(response.data) > 0:
                logger.info(f"✅ Firm created: {name}")
                return response.data[0]["id"]
            raise ValueError("Failed to create firm")
        except Exception as e:
            logger.error(f"Error creating firm: {e}")
            raise

    def set_firm_api_key(self, firm_id: str, api_key: str, provider: str) -> None:
        """Set API key for a firm via RPC."""
        try:
            self.client.rpc("set_firm_api_key", {
                "p_firm_id": firm_id,
                "p_api_key": api_key,
                "p_provider": provider
            }).execute()
            logger.info(f"🔑 {provider} key set")
        except Exception as e:
            logger.warning(f"⚠️ Could not set {provider} key: {e}")

    def get_or_create_campaign(self, firm_id: str, name: str, status: str = "draft") -> str:
        """Get existing campaign or create new one. Returns campaign_id."""
        try:
            response = self.client.table("campaigns")\
                .select("*")\
                .eq("firm_id", firm_id)\
                .eq("name", name)\
                .single()\
                .execute()
            
            if response.data:
                logger.info(f"ℹ️ Campaign already exists: {name}")
                return response.data["id"]
        except Exception as e:
             if "PGRST116" not in str(e):
                raise

        # Create new
        response = self.client.table("campaigns").insert({
            "firm_id": firm_id,
            "name": name,
            "status": status
        }).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"✅ Campaign created: {name}")
            return response.data[0]["id"]
        raise ValueError("Failed to create campaign")

    def upsert_prompt_template(self, campaign_id: str, name: str, action: str) -> None:
        """Upsert prompt template for a campaign."""
        try:
            self.client.table("campaigns_prompt_templates").upsert({
                "campaign_id": campaign_id,
                "prompt_template_name": name,
                "action": action
            }, on_conflict="prompt_template_name, campaign_id, action").execute()
            # logger.info(f"  - Processed prompt template: {name}")
        except Exception as e:
            logger.warning(f"⚠️ Error inserting prompt {name}: {e}")

    def ensure_user(self, email: str, firm_id: str, is_admin: bool = False) -> Optional[str]:
        """Ensure auth user and public user record exist. Returns user_id.
        
        If the user already exists in the public table, we just return their ID without
        trying to update their details (to avoid overwriting existing data).
        
        Args:
            email: User's email address
            firm_id: Firm ID to associate new users with
            is_admin: Whether the user should be an admin
        """
        user_id = None
        existing_public_user = False
        
        # 1. First check if user already exists in public table
        try:
            response = self.client.table("user").select("id, firm_id").eq("email", email).single().execute()
            if response.data:
                user_id = response.data["id"]
                existing_public_user = True
                logger.info(f"👤 User already exists: {email}")
                return user_id  # User exists, just return their ID
        except Exception as e:
            # PGRST116 means not found, which is expected for new users
            if "PGRST116" not in str(e):
                logger.warning(f"⚠️ Error checking user existence: {e}")
        
        # 2. User doesn't exist in public table, try to create auth user
        try:
            auth_response = self.client.auth.admin.create_user({
                "email": email,
                "email_confirm": True
            })
            if auth_response.user:
                user_id = auth_response.user.id
                logger.debug(f"Created new Auth User: {email}")
        except Exception as e:
            # Likely already exists in auth but not in public table
            pass

        # 3. If auth create failed, try to find in auth users
        if not user_id:
            try:
                users_response = self.client.auth.admin.list_users(page=1, per_page=100)
                users_list = users_response.users if hasattr(users_response, 'users') else users_response
                for u in users_list:
                    if u.email == email:
                        user_id = u.id
                        break
            except Exception as e:
                logger.error(f"Error listing users: {e}")
        
        if not user_id:
            logger.error(f"❌ Could not resolve User ID for {email}")
            return None

        # 4. Create public user record (only reached for new users)
        try:
            user_data = {
                "id": user_id,
                "firm_id": firm_id,
                "email": email,
                "status": "active",
                "is_admin": is_admin
            }
            
            self.client.table("user").insert(user_data).execute()
            logger.info(f"👤 Created new user: {email} (Firm: {firm_id}, Admin: {is_admin})")
        except Exception as e:
            logger.error(f"❌ Error creating public user {email}: {e}")
            return None
            
        return user_id

    def link_user_to_campaign(self, user_id: str, campaign_id: str) -> None:
        """Link a user to a campaign."""
        try:
            self.client.table("user_campaign").upsert({
                "user_id": user_id,
                "campaign_id": campaign_id
            }, on_conflict="user_id, campaign_id").execute()
        except Exception as e:
            logger.error(f"❌ Error linking user to campaign: {e}")

    def get_campaign_competitor_ids(self, campaign_id: str) -> List[Dict]:
        """Get competitor IDs and names for a campaign.

        Returns list of {"competitor_id": str, "name": str} dicts, or [] on error.
        """
        try:
            response = self.client.table("campaign_competitors")\
                .select("competitor_id, name")\
                .eq("campaign_id", campaign_id)\
                .execute()
            return response.data or []
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch competitors for campaign {campaign_id}: {e}")
            return []

    def get_competitor_description(self, campaign_id: str, competitor_id: str) -> Optional[str]:
        """Get a human-readable description for a specific competitor.

        Reads from research_data (jsonb) and renders to text.
        Falls back to description_md if research_data is not available.
        """
        try:
            response = self.client.table("campaign_competitors")\
                .select("research_data, description_md")\
                .eq("campaign_id", campaign_id)\
                .eq("competitor_id", competitor_id)\
                .single()\
                .execute()
            if response.data:
                research_data = response.data.get("research_data")
                if research_data:
                    from services.competitor_fetcher import render_competitor_description
                    return render_competitor_description(research_data)
                return response.data.get("description_md")
            return None
        except Exception as e:
            if "PGRST116" not in str(e):
                logger.warning(f"⚠️ Could not fetch competitor description for {competitor_id}: {e}")
            return None

    def upsert_competitors(self, campaign_id: str, competitors: list[dict]) -> int:
        """Replace all competitors for a campaign with fresh data.

        Deletes existing rows then inserts new ones.

        Args:
            campaign_id: The campaign to update.
            competitors: List of dicts with: competitor_id, name, aliases, research_data.

        Returns:
            Number of competitors inserted.
        """
        try:
            self.client.table("campaign_competitors")\
                .delete()\
                .eq("campaign_id", campaign_id)\
                .execute()
        except Exception as e:
            logger.warning(f"Error deleting old competitors for {campaign_id}: {e}")

        rows = []
        for comp in competitors:
            rows.append({
                "campaign_id": campaign_id,
                "competitor_id": comp["competitor_id"],
                "name": comp["name"],
                "aliases": comp.get("aliases", []),
                "research_data": comp.get("research_data", {}),
            })

        if rows:
            try:
                self.client.table("campaign_competitors").insert(rows).execute()
            except Exception as e:
                logger.error(f"Error inserting competitors for {campaign_id}: {e}")
                raise

        return len(rows)

    def get_competitor_research_data(self, campaign_id: str, competitor_id: str) -> Optional[dict]:
        """Get the structured research_data for a specific competitor."""
        try:
            response = self.client.table("campaign_competitors")\
                .select("research_data")\
                .eq("campaign_id", campaign_id)\
                .eq("competitor_id", competitor_id)\
                .single()\
                .execute()
            if response.data:
                return response.data.get("research_data")
            return None
        except Exception as e:
            if "PGRST116" not in str(e):
                logger.warning(f"Error fetching research data for {competitor_id}: {e}")
            return None
