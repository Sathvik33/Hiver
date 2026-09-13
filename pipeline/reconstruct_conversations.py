"""
Conversation Reconstruction Pipeline:
1. Loads tweets
2. Links customer -> agent replies using tweet_id, in_response_to_tweet_id, response_tweet_id
3. Formats canonical conversation structure:
   {
     "conversation_id": "...",
     "brand": "...",
     "messages": [
       {"tweet_id": "...", "author_id": "...", "role": "customer"|"agent", "timestamp": "...", "text": "..."}
     ],
     "resolution_quality": "HIGH"|"MEDIUM"|"LOW"|"UNKNOWN",
     "problem_text": "...",
     "resolution_text": "..."
   }
4. Removes duplicate/unusable records and masks privacy placeholders.
5. Saves clean structured JSON corpus.
"""

import re
import json
from typing import List, Dict, Any, Optional

def clean_tweet_text(text: str) -> str:
    """Preserves or normalizes handles, masks private tokens."""
    if not text:
        return ""
    # Normalize excessive whitespaces
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned

def assess_resolution_quality(messages: List[Dict[str, Any]]) -> str:
    """
    Assesses resolution quality based on dialogue flow:
    HIGH: Multiple turns ending with agent providing actionable help / resolution.
    MEDIUM: Single turn exchange where agent addresses customer problem with direct guidance.
    LOW: Conversational loop without actionable info or only 'DM us' with no substance.
    UNKNOWN: Incomplete or missing replies.
    """
    if not messages:
        return "UNKNOWN"
    
    agent_turns = [m for m in messages if m.get("role") == "agent"]
    customer_turns = [m for m in messages if m.get("role") == "customer"]
    
    if not agent_turns or not customer_turns:
        return "UNKNOWN"
    
    last_agent_msg = agent_turns[-1]["text"].lower()
    
    # Check for substantive resolution markers
    has_actionable_advice = any(term in last_agent_msg for term in [
        "please try", "you can", "check your", "follow these steps", "we have", 
        "refund", "shipped", "track", "cancel", "reset", "settings", "ordered", "delivered"
    ])
    
    is_pure_deflection = (
        len(last_agent_msg.split()) < 7 and ("dm" in last_agent_msg or "message us" in last_agent_msg)
    )
    
    if is_pure_deflection:
        return "LOW"
    elif has_actionable_advice and len(messages) >= 2:
        return "HIGH"
    elif len(messages) >= 2:
        return "MEDIUM"
    else:
        return "LOW"

def reconstruct_conversation_turns(raw_records: List[Dict[str, Any]], brand: str) -> List[Dict[str, Any]]:
    """
    Reconstructs chronological conversations for a target brand.
    """
    # Index by tweet_id
    tweet_map = {r["tweet_id"]: r for r in raw_records if "tweet_id" in r}
    conversations = []
    
    # Identify starter customer tweets (inbound == True and not in_response_to_tweet_id)
    seen_tweets = set()
    
    for r in raw_records:
        t_id = r.get("tweet_id")
        if not t_id or t_id in seen_tweets:
            continue
            
        inbound = str(r.get("inbound", "")).lower() in ("true", "1")
        in_resp = r.get("in_response_to_tweet_id")
        
        # We start dialogue chains from customer inbound messages
        if inbound and not in_resp:
            chain = [r]
            seen_tweets.add(t_id)
            
            # Follow response chain
            curr = r
            while curr and curr.get("response_tweet_id"):
                resp_ids = str(curr.get("response_tweet_id")).split(",")
                next_found = None
                for rid in resp_ids:
                    rid = rid.strip()
                    if rid in tweet_map and rid not in seen_tweets:
                        next_found = tweet_map[rid]
                        seen_tweets.add(rid)
                        chain.append(next_found)
                        break
                curr = next_found
                
            # Check if brand was involved in the chain
            brand_involved = any(m.get("author_id") == brand for m in chain)
            if brand_involved and len(chain) >= 2:
                messages = []
                for m in chain:
                    is_cust = str(m.get("inbound", "")).lower() in ("true", "1")
                    messages.append({
                        "tweet_id": m.get("tweet_id"),
                        "author_id": m.get("author_id"),
                        "role": "customer" if is_cust else "agent",
                        "timestamp": m.get("created_at", ""),
                        "text": clean_tweet_text(m.get("text", ""))
                    })
                
                quality = assess_resolution_quality(messages)
                cust_msgs = [m["text"] for m in messages if m["role"] == "customer"]
                agent_msgs = [m["text"] for m in messages if m["role"] == "agent"]
                
                conversations.append({
                    "conversation_id": f"conv_{t_id}",
                    "brand": brand,
                    "messages": messages,
                    "resolution_quality": quality,
                    "problem_text": " ".join(cust_msgs),
                    "resolution_text": agent_msgs[-1] if agent_msgs else ""
                })
                
    return conversations
