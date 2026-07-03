import sys
import time
import random
import re  

# Expanded Knowledge Base
INTENTS = {
    "greeting": {
        "keywords": [r"\bhi\b", r"\bhello\b", r"\bhey\b", r"\bgood morning\b", r"\bsalam\b", r"\baoa\b"],
        "responses": ["Hello! Welcome to our customer support. How can I help you today?", 
                      "Hi there! How can I assist you right now?"]
    },
    "bot_identity": {
        "keywords": [r"\bwho are you\b", r"\btum kon ho\b", r"\bkaun ho\b", r"\bidentity\b", r"\bwhat are you\b"],
        "responses": ["I am an AI-powered customer support virtual assistant.", 
                      "Main ek AI support chatbot hoon, aapki madad ke liye banaya gaya hoon!"]
    },
    "support_hours": {
        "keywords": [r"\bhours\b", r"\btiming\b", r"\bopen\b", r"\bclose\b", r"\btime\b", r"\btiming kya hai\b"],
        "responses": ["Our support team is available 24/7 online!", 
                      "We are open Monday to Sunday, 24 hours a day to assist you."]
    },
    "refund_policy": {
        "keywords": [r"\brefund\b", r"\breturn\b", r"\bmoney back\b", r"\bpaise wapas\b", r"\bcancel order\b"],
        "responses": ["Our refund policy allows you to request a full refund within 14 days of purchase.", 
                      "Aap 14 din ke andar refund claim kar sakte hain. 100% money-back guarantee hai."]
    },
    "contact_human": {
        "keywords": [r"\bagent\b", r"\bhuman\b", r"\brepresentative\b", r"\bcall\b", r"\bspeak to someone\b", r"\bbaat karwao\b"],
        "responses": ["I can connect you to a live human agent. Please hold on...", 
                      "Sure, transferring your chat to a support representative. One moment please."]
    },
    "goodbye": {
        "keywords": [r"\bbye\b", r"\bgoodbye\b", r"\bexit\b", r"\bquit\b", r"\bthanks\b", r"\bthank you\b"],
        "responses": ["Thank you for contacting us. Have a great day!", 
                      "Goodbye! Feel free to reach out if you need anything else."]
    }
}

def get_chatbot_response(user_input):
    user_input = user_input.lower().strip()
    
    # Advanced RegEx Matching Logic (Tries to find whole words)
    for intent, data in INTENTS.items():
        for pattern in data["keywords"]:
            if re.search(pattern, user_input):
                return random.choice(data["responses"]), intent
                
    # Fallback if chatbot doesn't understand
    return "I'm sorry, I didn't quite understand that. You can ask about our 'refund policy', 'timings', 'who I am', or type 'agent' to speak to a human.", "unknown"

def main():
    print("="*50)
    print("          AI CUSTOMER SUPPORT CHATBOT          ")
    print("="*50)
    print("Bot: Hello! I am your virtual assistant. Type 'exit' or 'bye' to stop the chat.\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            response, intent = get_chatbot_response('bye')
            print(f"Bot: {response}")
            break
            
        response, intent = get_chatbot_response(user_input)
        
        print("Bot is typing...", end="\r")
        time.sleep(0.4)
        
        print(f"Bot: {response}\n")

if __name__ == "__main__":
    main()