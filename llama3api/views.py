# chatbot/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ChatbotQuerySerializer,QuerySerializer
from .llama3file import ChatbotModel
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os
# from .llama3file import initialize_components

class ChatbotView(APIView):
    def post(self, request):
        serializer = ChatbotQuerySerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query']
            chatbot = ChatbotModel()
            response = chatbot.generate_response(query)
            return Response({"response": response}, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)



class QueryView(APIView):
    def post(self, request):
        serializer = QuerySerializer(data=request.data)
        if serializer.is_valid():
            question = serializer.validated_data['question']
            user_prompt = request.data.get('sys_prompt')
            botname = request.data.get('botname', 'Chatbot:')
            file = request.FILES.get('file')  # Ensure you get the file from the request

            if not file:
                return Response({"error": "No file was submitted."}, status=status.HTTP_400_BAD_REQUEST)

            # Save the uploaded file
            file_name = default_storage.save(file.name, ContentFile(file.read()))
            file_path = default_storage.path(file_name)

            # Initialize LangChain components with the file path
            chatbot = ChatbotModel()
            qa_chain = chatbot.initialize_components(file_path,question,user_prompt,botname)

            # Run the query through the chain
            # result = qa_chain({"query": question})

            # Clean up the uploaded file after processing
            os.remove(file_path)

            return Response(qa_chain, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)