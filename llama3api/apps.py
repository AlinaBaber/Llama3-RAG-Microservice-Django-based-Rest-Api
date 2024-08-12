from django.apps import AppConfig
from . import model_loader 

class Llama3ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'llama3api'

    def ready(self):
        #from . import model_loader  # Import a module where you handle model loading
        model_loader.load_all_models()  # Function to load models
