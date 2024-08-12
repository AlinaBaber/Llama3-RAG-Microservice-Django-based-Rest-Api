# from django.urls import path, include
# from rest_framework.routers import DefaultRouter
# from .views import BookViewSet
#
# router = DefaultRouter()
# router.register(r'books', BookViewSet)
#
# urlpatterns = [
#     path('', include(router.urls)),
# ]


from django.urls import path
from .views import ChatbotView, QueryView

urlpatterns = [
    path('chatbot/', ChatbotView.as_view(), name='chatbot'),
    path('query/', QueryView.as_view(), name='query'),
]