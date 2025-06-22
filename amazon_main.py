import os
import uuid
import boto3
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from updated_universal_service_bot import UniversalServiceBot  # Your universal bot
from dotenv import load_dotenv
import ngrok
from twilio.rest import Client
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse,Response, FileResponse,JSONResponse
from fastapi.exceptions import HTTPException
import uvicorn
import sys, logging, io
from twilio.twiml.messaging_response import MessagingResponse
from fastapi.middleware.cors import CORSMiddleware
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant

load_dotenv()  # Loads .env if present
PORT = int(os.getenv('PORT', 5003))  # Default to 5003 if not set
sessions = {}

app = FastAPI()
# --- Add CORS Middleware ---
# Define the origins that are allowed to make cross-site requests.
# It's crucial to include the origins from which your Capacitor app will be served.
origins = [
    "https://orderlybite.com",        # Your production frontend (Vercel)
    "https://www.orderlybite.com",    # Your production frontend with www
    "https://omniassistai.com",        # Your production frontend (Vercel)
    "https://www.omniassistai.com",    # Your production frontend with www
    # Origins for Capacitor apps [6]
    "capacitor://localhost",        # For Capacitor iOS local scheme
    "ionic://localhost",            # Another common Capacitor local scheme (though less used with raw Capacitor)
    "http://localhost",             # For Capacitor Android local scheme AND potentially some iOS WKWebView scenarios if not using a custom scheme

    # Origins for local development servers
    "http://localhost:5173",        # Common Vite dev server port (if you use live reload: ionic cap run ios -l)
    "http://localhost:8100",        # Common Ionic serve port (if you use live reload: ionic cap run ios -l)
    
    # Your previously listed local dev server ports
    "http://localhost:8080",        
    "http://localhost:5003",        
    "http://localhost:5006",  
    "http://localhost:5007",

    # It's also good practice to allow your backend's own origin if it ever serves a frontend
    "https://api.orderlybite.com",
    "https://api.omniassistai.com" 
]
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)  

polly = boto3.client("polly", region_name="us-east-1")
TWILIO_NUMBER = os.getenv('TWILIO_NUMBER')
authtoken = os.getenv("NGROK_AUTHTOKEN")
ngrok.set_auth_token(authtoken)


# Dictionary to hold session data
data_sessions = {}
chat_sessions = {}
active_calls = {}

def update_twilio_webhook(ngrok_url, webhook_type):
    """
    Updates either voice or SMS webhook for a Twilio phone number.
    
    Args:
        ngrok_url (str): The base ngrok URL
        webhook_type (str): Either 'voice' or 'sms'
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get credentials from environment variables
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    phone_number = os.getenv("TWILIO_VOICE_NUMBER")
    
    print(f"Using account SID: {account_sid}")
    
    # Initialize Twilio client
    client = Client(account_sid, auth_token)
    
    try:
        # Find the phone number
        numbers = client.incoming_phone_numbers.list(phone_number=phone_number)
        if not numbers:
            print(f"No phone number found matching {phone_number}")
            return False
        
        number_sid = numbers[0].sid
        
        # Update based on webhook type
        if webhook_type.lower() == 'voice':
            client.incoming_phone_numbers(number_sid).update(
                voice_url=f"{ngrok_url}/voice",
                status_callback=f"{ngrok_url}/status",
                status_callback_method="POST"
            )
            print(f"Updated voice webhook for {phone_number} to {ngrok_url}/voice")
        
        elif webhook_type.lower() == 'sms':
            client.incoming_phone_numbers(number_sid).update(
                sms_url=f"{ngrok_url}/sms",
                sms_method="POST"
            )
            print(f"Updated SMS webhook for {phone_number} to {ngrok_url}/sms")
        
        else:
            print(f"Invalid webhook type: {webhook_type}. Use 'voice' or 'sms'.")
            return False
        
        return True
        
    except Exception as e:
        print(f"Failed to update {webhook_type} webhook URL: {str(e)}")
        return False


@app.post("/status")
async def status(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid")
    call_status = form.get("CallStatus")
    print(f"Status callback: CallSid={call_sid}, CallStatus={call_status}")

    # Clean up session when call is completed
    if call_status == "completed" and call_sid in sessions:
        sessions.pop(call_sid, None)
        print(f"Cleaned up session for CallSid={call_sid}")
    return Response(status_code=204)
  
# --- 3. Polly synthesis ---
def synthesize_with_polly(text, voice_id="Joanna"):
    polly_response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voice_id,
        Engine="neural"
    )
    audio_filename = f"/tmp/{uuid.uuid4()}.mp3"
    with open(audio_filename, "wb") as f:
        f.write(polly_response["AudioStream"].read())
    return audio_filename


@app.api_route("/sms", methods=["GET", "POST"])
async def sms_reply(request: Request):
    global CALLER_ID
    
    print(f"🔍 SMS endpoint called - Method: {request.method}")
    
    # Handle both form data and JSON data
    try:
        # Try form data first (for Twilio webhooks and curl)
        form_data = await request.form()
        message_body = form_data.get("Body", "").strip()
        CALLER_ID = form_data.get("From", "").replace("whatsapp:", "")
        print(f"✅ Form data - Body: '{message_body}', From: '{CALLER_ID}'")
    except Exception as form_error:
        print(f"⚠️ Form parsing failed, trying JSON: {form_error}")
        # Fallback to JSON data (for frontend requests)
        try:
            json_data = await request.json()
            message_body = json_data.get("message", "").strip()
            CALLER_ID = json_data.get("from", "frontend_user")
            print(f"✅ JSON data - message: '{message_body}', from: '{CALLER_ID}'")
        except Exception as json_error:
            print(f"❌ Both form and JSON parsing failed: {json_error}")
            message_body = ""
            CALLER_ID = "frontend_user"
    
    # Ensure CALLER_ID is not empty or "UNKNOWN"
    if not CALLER_ID or CALLER_ID == "UNKNOWN":
        old_caller_id = CALLER_ID
        CALLER_ID = "frontend_user"
        print(f"🔄 Changed CALLER_ID from '{old_caller_id}' to '{CALLER_ID}'")
    
    # Always ensure session exists
    if CALLER_ID not in chat_sessions:
        print(f"🔍 Creating new session for '{CALLER_ID}'...")
        try:
            chat_sessions[CALLER_ID] = UniversalServiceBot("./sectors",
                                                          "universal_chroma_database", 
                                                          CALLER_ID)
            print(f"✅ Session created for '{CALLER_ID}'")
        except Exception as e:
            print(f"❌ Failed to create session for '{CALLER_ID}': {e}")
            response = MessagingResponse()
            response.message("I'm here to help with your request. Could you tell me more about what you need?")
            return HTMLResponse(content=str(response), media_type="application/xml")

    # Initialize data session if not already started
    if CALLER_ID not in data_sessions:
        data_sessions[CALLER_ID] = []

    # Handle exit command
    if message_body.lower() == "exit":
        if CALLER_ID in data_sessions:
            del data_sessions[CALLER_ID]
        if CALLER_ID in chat_sessions:
            del chat_sessions[CALLER_ID]
        response = MessagingResponse()
        response.message("Session ended. Goodbye!")
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Save the message to the session
    data_sessions[CALLER_ID].append(message_body)

    # Generate a response with error handling
    response = MessagingResponse()
    try:
        print(f"🔍 Processing message for '{CALLER_ID}': '{message_body}'")
        chatResponse = chat_sessions[CALLER_ID].chatAway(message_body)
        response.message(str(chatResponse))
        print("CHATBOT: {}".format(str(chatResponse)))
    except Exception as e:
        print(f"❌ chatAway error for '{CALLER_ID}': {e}")
        response.message("I'm here to help with your request. Could you tell me more about what you need?")

    print("DEBUG: Send following to caller: {}".format(response.to_xml()))
    return HTMLResponse(content=response.to_xml(), media_type="application/xml")
   


#@app.api_route("/sms2", methods=["GET", "POST"])
async def sms_reply2(request: Request):
    global CALLER_ID

    # Parse form data for POST request
    form_data = await request.form()
    message_body = form_data.get("Body", "").strip()
    CALLER_ID = form_data.get("From", "").replace("whatsapp:","")

    # Handle missing caller ID
    if not CALLER_ID:
        CALLER_ID = "UNKNOWN"
        print("DEBUG: Caller ID could not be retrieved.")
    
    # Always ensure session exists
    if CALLER_ID not in chat_sessions:
        print(f"DEBUG: Creating new session for {CALLER_ID}.")
        chat_sessions[CALLER_ID] = UniversalServiceBot("./sectors",
                                                      "universal_chroma_database",CALLER_ID)

    # Initialize session if not already started
    if CALLER_ID not in data_sessions:
        data_sessions[CALLER_ID] = []

    # Handle exit command
    if message_body.lower() == "exit":
        del data_sessions[CALLER_ID]
        response = MessagingResponse()
        response.message("Session ended. Goodbye!")
        return HTMLResponse(content=str(response), media_type="application/xml")

    # Save the message to the session
    data_sessions[CALLER_ID].append(message_body)

    # Generate a response
    response = MessagingResponse()
    chatResponse = chat_sessions[CALLER_ID].chatAway(message_body)
    response.message(str(chatResponse))
    print("CHATBOT: {}".format(str(chatResponse)))

    print("DEBUG: Send following to caller: {}".format(response.to_xml()))
    return HTMLResponse(content=response.to_xml(), media_type="application/xml")


# --- 4. Flask routes ---
@app.post("/voice")
async def voice(request: Request):
    try:
        form = await request.form()
        phone_number_called = form.get("To")
        transcription = form.get("SpeechResult", "")
        
        call_sid = form.get("CallSid") or phone_number_called

        if call_sid not in sessions:
            sessions[call_sid] = UniversalServiceBot("./sectors", 
                                                     "universal_chroma_database", 
                                                     phone_number_called)
            sessions[call_sid].prompt_count = 0
        py = sessions[call_sid]
        py.prompt_count += 1

        print(f"Call to: {phone_number_called}, User said: {transcription} call_sid:{call_sid}")
        response = VoiceResponse()
        if not transcription:
            gather = Gather(
                input="speech",
                action="/voice",
                method="POST",
                timeout=5,
                speech_timeout="auto",
                barge_in=True   # Enable barge-in!
            )
            if py.prompt_count == 1:
                # Initial prompt
                gather.say("Welcome Melville Deli! Please tell me your order.", voice="Polly.Joanna-Neural",language="en-US")
            else:
                # Repeat prompt or different message
                gather.say("Sorry, I didn't hear anything. Please tell me your order.", voice="Polly.Joanna-Neural",language="en-US")

            response.append(gather)
            response.redirect("/voice")
            return Response(str(response), media_type="application/xml")
    
       
        # 2. Call chatService with transcription
        response_text = py.chatAway(transcription)
        print("Bot response:", response_text)

        # 5. Optionally, gather next user input for multi-turn dialog
        # Next turn: respond and gather more input
        gather = Gather(
            input="speech",
            action="/voice",
            method="POST",
            timeout=5,
            barge_in=True
        )
        gather.say(response_text,voice="Polly.Joanna-Neural",language="en-US")
        response.append(gather)
        response.redirect("/voice")
        #return JSONResponse(content={"message": str(response)}) 
        return Response(str(response), media_type="application/xml")
    except Exception as e:
        logging.error(f"Error handling voice call: {str(e)}")
        response = VoiceResponse()
        response.say("We're sorry, but there was an error processing your call.")
        response.hangup()
        #return JSONResponse(content={"message": str(response)}) 
        return Response(content=str(response), media_type="application/xml")

@app.post('/token')
async def token():
    # Your environment variables are correct
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    api_key = os.getenv('TWILIO_API_KEY_SID')
    api_secret = os.getenv('TWILIO_API_SECRET')
    twiml_app_sid = os.getenv('TWIML_APP_SID')
    
    # Create access token with voice grant
    access_token = AccessToken(account_sid, api_key, api_secret, identity='user123')
    
    # Create voice grant
    voice_grant = VoiceGrant(
        outgoing_application_sid=twiml_app_sid,
        incoming_allow=True
    )
    
    access_token.add_grant(voice_grant)
    
    # Use FastAPI's JSONResponse instead of Flask's jsonify
    return JSONResponse(content={'token': access_token.to_jwt()})

@app.post('/voice-status')
async def voice_status(request: Request):
    form_data = await request.form()
    call_status = form_data.get("CallStatus")
    print(f"Voice call status: {call_status}")
    return JSONResponse(content={'msg':"OK"})

if __name__ == "__main__":

    # Open ngrok tunnel
    listener = ngrok.forward(f"http://localhost:{PORT}")
    print(f"Ngrok tunnel opened at {listener.url()} for port {PORT}")
    NGROK_URL = listener.url()
    update_twilio_webhook(NGROK_URL, "voice")
    update_twilio_webhook(NGROK_URL, "sms")

    uvicorn.run(app, host="0.0.0.0", 
                port=PORT,
                ws_ping_interval=60,  # Increase from default 20 seconds
                ws_ping_timeout=30 ) 
    

