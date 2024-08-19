import threading
import record
import face2

hat = False
glasses = True
beard = False

# Add functions to thread to run in parallel
video_thread = threading.Thread(target=face2.display_video)
audio_thread = threading.Thread(target=record.record_audio)

# Run threads
video_thread.start()
audio_thread.start()

# Wait for threads to finish
video_thread.join()
audio_thread.join()
