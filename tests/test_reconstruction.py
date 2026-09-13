import pytest
from pipeline.reconstruct_conversations import reconstruct_conversation_turns, assess_resolution_quality, clean_tweet_text

def test_clean_tweet_text():
    raw = "  @AmazonHelp   my   package is missing!  "
    assert clean_tweet_text(raw) == "@AmazonHelp my package is missing!"

def test_assess_resolution_quality_high():
    msgs = [
        {"role": "customer", "text": "Where is my item?"},
        {"role": "agent", "text": "Please check your tracking link or please try track via amzn.to/track"}
    ]
    assert assess_resolution_quality(msgs) == "HIGH"

def test_assess_resolution_quality_low_deflection():
    msgs = [
        {"role": "customer", "text": "Help"},
        {"role": "agent", "text": "Please DM us"}
    ]
    assert assess_resolution_quality(msgs) == "LOW"

def test_conversation_reconstruction():
    raw_tweets = [
        {"tweet_id": "1", "author_id": "user1", "inbound": "True", "text": "Package late", "response_tweet_id": "2"},
        {"tweet_id": "2", "author_id": "AmazonHelp", "inbound": "False", "text": "We apologize. Please try tracking at amzn.to/track", "in_response_to_tweet_id": "1"}
    ]
    convs = reconstruct_conversation_turns(raw_tweets, "AmazonHelp")
    assert len(convs) == 1
    assert convs[0]["conversation_id"] == "conv_1"
    assert len(convs[0]["messages"]) == 2
    assert convs[0]["resolution_quality"] == "HIGH"
