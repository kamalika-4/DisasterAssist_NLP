"""
Speech Processing module.
- Speech-to-Text: Google Speech Recognition via SpeechRecognition
  on uploaded/recorded audio.
- Text-to-Speech: gTTS-based synthesis for playback of guidance
  (falls back gracefully with a clear message if network TTS is
  unavailable in the runtime environment).
- Audio feature extraction (duration, sample rate, RMS energy) shown
  in the UI as real signal-processing output.
"""
import io
import os
import wave
import audioop
import tempfile
import speech_recognition as sr


def transcribe_audio_bytes(audio_bytes: bytes, filename_hint: str = "audio.wav", language: str = "en-US"):
    """Transcribe raw audio bytes using Google Speech Recognition."""
    r = sr.Recognizer()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename_hint)[1] or ".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with sr.AudioFile(tmp_path) as source:
            audio_data = r.record(source)

        try:
            text = r.recognize_google(audio_data, language=language)
            engine = "Google Speech Recognition"
        except sr.UnknownValueError:
            return {"success": False, "error": "Could not understand audio (no clear speech detected).", "engine": None}
        except sr.RequestError as e:
            return {"success": False, "error": f"Speech recognition service error: {e}", "engine": None}
        except Exception as e:
            return {"success": False, "error": f"Recognition engine error: {e}", "engine": None}

        return {"success": True, "text": text, "engine": engine}
    except Exception as e:
        return {"success": False, "error": f"Could not process audio file: {e}", "engine": None}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def analyze_audio_signal(audio_bytes: bytes):
    """Compute real basic signal statistics from a WAV file."""
    try:
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / float(frame_rate) if frame_rate else 0
            raw = wf.readframes(n_frames)
            rms = audioop.rms(raw, sample_width) if raw else 0
        return {
            "channels": n_channels,
            "sample_rate": frame_rate,
            "duration_sec": round(duration, 2),
            "rms_energy": rms,
            "n_frames": n_frames,
        }
    except Exception as e:
        return {"error": str(e)}


def synthesize_speech(text: str, lang: str = "en"):
    """Attempt gTTS synthesis; returns bytes or an error message.
    Requires outbound internet to Google TTS endpoint -- may be
    unavailable in sandboxed/offline environments."""
    try:
        from gtts import gTTS
        gtts_lang_map = {"en": "en", "ta": "ta", "hi": "hi", "es": "es", "fr": "fr"}
        tts = gTTS(text=text, lang=gtts_lang_map.get(lang, "en"))
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return {"success": True, "audio_bytes": buf.read()}
    except Exception as e:
        return {"success": False, "error": str(e)}
