import subprocess
import time
from datetime import datetime
import os
import base64
import openai
import requests
from dotenv import load_dotenv
import pygame
import cv2
from collections import deque

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ElevenLabs API key
elevenlabs_api_key = os.getenv("ELEVENLABS_KEY")

# Initialize pygame mixer
pygame.mixer.init()

# Directory to store audio files
AUDIO_DIR = "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create the main photos directory if it doesn't exist
PHOTOS_DIR = "photos"
os.makedirs(PHOTOS_DIR, exist_ok=True)

INTERVAL = 5  # or whatever interval you want between photos

class ImageAnalyzer:
    def __init__(self, client):
        self.message_history = deque(maxlen=10)
        self.client = client

    def analyze_image(self, image_path):
        # Read and encode the image
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')


        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image detailed. Pay extra attention to the people \
                        in the image and what they are doing."},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        description = response.choices[0].message.content
        
        messages = [
            {"role": "system", "content": "You are two skeletons named Knotan and Skallan, commenting on scenes in a garden during Halloween."}
        ]
        
        #messages.extend(list(self.message_history))
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Imagine that you are two skeletons (Knotan och Skallan) in a garden\
						during halloween and you are watching the scene described below. \
						let the skeletons comment what they see and call on the people they are abnalyuzing like: you therein the green hoodie.\
						as in the style of Statler and Waldorf in the muppets but keep it short and catchy and funny and in swedish max four sentences? never use markup. \
                        Each line xtarst with the ane of the skelleton talking  like Skallan: \
                      Here is the description of the scene: " + description}
            ],
        })

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
        )

        dialog = response.choices[0].message.content

        self.message_history.append({"role": "assistant", "content": dialog})

        return dialog

def setup_camera():
    """List available cameras and let the user select one."""
    result = subprocess.run(["imagesnap", "-l"], capture_output=True, text=True)
    cameras = result.stdout.strip().split('\n')[1:]  # Skip the header line
    
    # Clean up camera names
    cameras = [camera.split('=>')[-1].strip() for camera in cameras]
    
    print("Available cameras:")
    for i, camera in enumerate(cameras):
        print(f"{i + 1}. {camera}")
    
    selection = int(input("Select a camera (enter the number): ")) - 1
    return cameras[selection]

def create_photo_folder():
    """Create a new subfolder for the current photo."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_folder = os.path.join(PHOTOS_DIR, timestamp)
    os.makedirs(photo_folder, exist_ok=True)
    return photo_folder

def take_photo(camera):
    """Take a photo and save it in a new subfolder."""
    photo_folder = create_photo_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(photo_folder, f"photo_{timestamp}.jpg")
    
    result = subprocess.run(["imagesnap", "-d", camera, "-w", "2", filename], capture_output=True, text=True)
    
    if os.path.exists(filename):
        print(f"Photo saved as {filename}")
        if has_people(filename):
            return filename, photo_folder
        else:
            print("No people detected in the image. Skipping analysis.")
            os.remove(filename)
            os.rmdir(photo_folder)
            return None, None
    else:
        print(f"Error: Could not capture image. {result.stderr}")
        os.rmdir(photo_folder)
        return None, None

def generate_audio(text, voice_id, photo_folder):
    """Generate audio and save it in the photo folder."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "language_code": "en",  # Changed to English for this example
        "voice_settings": {
            "stability": 0.31,
            "similarity_boost": 0.97,
            "style": 0.50,
            "use_speaker_boost": True
        },
        "seed": 123,
    }
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": elevenlabs_api_key
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        audio_filename = os.path.join(photo_folder, f"audio_{int(time.time())}.mp3")
        with open(audio_filename, "wb") as f:
            f.write(response.content)
        print(f"Created new audio file: {audio_filename}")
        return audio_filename
    else:
        print(f"Error generating audio: {response.status_code}, {response.text}")
        return None

def play_audio(audio_file):
    """Play the generated audio file."""
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def process_dialog(dialog, photo_folder):
    """Process the dialog, generate voices, and play them."""
    lines = dialog.strip().split('\n')
    
    knotan_voice = "f2yUVfK5jdm78zlpcZ8C"
    skallan_voice = "9yzdeviXkFddZ4Oz8Mok"
    
    for line in lines:
        if line.startswith("Knotan:"):
            voice_id = knotan_voice
            text = line.replace("Knotan:", "").strip()
        elif line.startswith("Skallan:"):
            voice_id = skallan_voice
            text = line.replace("Skallan:", "").strip()
        else:
            continue  # Skip lines that don't start with either prefix
        
        print(f"Processing line: {text}")
        
        audio_file = generate_audio(text, voice_id, photo_folder)
        if audio_file:
            play_audio(audio_file)
        
        time.sleep(0.5)

def has_people(image_path):
    """Check if the image contains people using face detection."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def main():
    camera = setup_camera()
    analyzer = ImageAnalyzer(client)  # Pass the OpenAI client to the analyzer

    while True:
        photo_path, photo_folder = take_photo(camera)
        print (photo_path)
        if photo_path:
            print ("ok)")
            dialog = analyzer.analyze_image(photo_path)
            print (dialog)
            if dialog:
                print ("ok2")
                process_dialog(dialog, photo_folder)
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    main()
