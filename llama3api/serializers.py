# myapp/serializers.py

from rest_framework import serializers


class ChatbotQuerySerializer(serializers.Serializer):
    query = serializers.CharField()


class QuerySerializer(serializers.Serializer):
    question = serializers.CharField()
    file = serializers.FileField()
    sys_prompt = serializers.CharField(required=False, allow_blank=True)
    botname = serializers.CharField(required=False, allow_blank=True)