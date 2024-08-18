import threading
import record
import face2

hat = False
glasses = True
beard = False

video_thread = threading.Thread(target=face2.display_video)
audio_thread = threading.Thread(target=record.record_audio)

# Chạy các luồng
video_thread.start()
audio_thread.start()

# Chờ các luồng kết thúc
video_thread.join()
audio_thread.join()

# 