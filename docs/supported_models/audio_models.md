# Audio Models

SGLang provides a lightweight TTS server for Qwen3-TTS models. The `sglang serve`
command automatically detects Qwen3-TTS checkpoints and launches the TTS server.

## Supported models

| Model Family | Example HuggingFace Identifier | Notes |
|--------------|--------------------------------|-------|
| **Qwen3-TTS** (12Hz 0.6B/1.7B) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | CustomVoice, VoiceDesign, and Base (voice clone) variants. |

## Serving

Start the server:

```bash
pip install "sglang[audio]"
sglang serve --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --host 0.0.0.0 --port 30000
```

Generate speech (CustomVoice):

```bash
curl -X POST http://localhost:30000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello, nice to meet you.",
    "language": "English",
    "speaker": "Vivian",
    "instruct": "Warm and friendly tone",
    "response_format": "wav"
  }' --output output.wav
```

VoiceDesign:

```bash
curl -X POST http://localhost:30000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "input": "We can start whenever you are ready.",
    "language": "English",
    "mode": "voice_design",
    "instruct": "Calm, mid-aged male, steady pace",
    "response_format": "wav"
  }' --output design.wav
```

VoiceClone (Base):

```bash
curl -X POST http://localhost:30000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "This is a cloned voice test.",
    "mode": "voice_clone",
    "language": "English",
    "ref_audio": "https://example.com/ref.wav",
    "ref_text": "Hello, this is a reference clip.",
    "response_format": "wav"
  }' --output clone.wav
```

Notes:
- `response_format` supports `wav` and `flac`.
- Streaming is not supported in the current TTS server.
