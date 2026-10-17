"""Emotion detection using IBM Watson NLP."""

import json
import requests


def emotion_detector(text_to_analyze):
    """Analyze text and return emotion scores."""
    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {"raw_document": {"text": text_to_analyze}}

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=30
    )

    if response.status_code == 400:
        return {
            "anger": None,
            "disgust": None,
            "fear": None,
            "joy": None,
            "sadness": None,
            "dominant_emotion": None
        }

    response.raise_for_status()
    data = json.loads(response.text)
    emotions = data["emotionPredictions"][0]["emotion"]

    result = {
        emotion: emotions[emotion]
        for emotion in ("anger", "disgust", "fear", "joy", "sadness")
    }
    result["dominant_emotion"] = max(result, key=result.get)

    return result
