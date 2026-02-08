import websocket
import wave
import time
import json

WS_URL = "ws://127.0.0.1:8000/ws/transcribe"
CHUNK_DURATION = 0.5
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

def stream_wav(file_path):
    print("🔌 Connecting to:", WS_URL)
    ws = websocket.WebSocket()
    ws.connect(WS_URL)
    ws.settimeout(0.2)   # 🔑 IMPORTANT
    print("Connected")

    with wave.open(file_path, "rb") as wf:
        print("🎧 Audio info:")
        print("  Channels:", wf.getnchannels())
        print("  Sample rate:", wf.getframerate())

        while True:
            frames = wf.readframes(CHUNK_SIZE)
            if not frames:
                print("End of audio")
                ws.send_text(json.dumps({"event": "end"}))
                while True:
                    try:
                        response = ws.recv()
                        data = json.loads(response)

                        # if "partial" in data:
                        #     print("📝 Partial:", data["partial"])

                        if "final" in data:
                            print("📝 Final:", data["final"])
                            break

                    except websocket._exceptions.WebSocketTimeoutException:
                        # Server still processing final transcription
                        continue
                break
                

            ws.send_binary(frames)
            print("➡ Sent chunk")

            time.sleep(CHUNK_DURATION)

            try:
                response = ws.recv()
                data = json.loads(response)
                if "partial" in data:
                    print("📝 Partial:", data["partial"])
            except websocket._exceptions.WebSocketTimeoutException:
                # No response yet — this is NORMAL
                pass

    ws.close()
    print("Connection closed")

if __name__ == "__main__":
    stream_wav("test.wav")
