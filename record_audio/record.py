import speech_recognition as sr
import pyttsx3 
import filter as filter

r = sr.Recognizer() 

def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command) 
    engine.runAndWait()

# Record function to activate filter
def record_audio():    
    while(1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.2)
                audio2 = r.listen(source2)

                MyText = r.recognize_google(audio2)
                MyText = MyText.lower()

                print("Did you say ",MyText)
                
                if "mangekyou" in MyText:
                    filter.mangekyou = True
                    filter.sharingan = True
                    filter.pic = 0
                elif "sharingan" in MyText:
                    filter.sharingan = True
                    filter.mangekyou = False
                    filter.pic = 0
                elif "close" in MyText:
                    filter.sharingan = False
                    filter.mangekyou = False
                
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            
        except sr.UnknownValueError:
            print("unknown error occurred")
