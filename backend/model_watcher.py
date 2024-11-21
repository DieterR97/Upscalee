from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
from pathlib import Path
import time
from flask_socketio import SocketIO

class ModelDirectoryHandler(FileSystemEventHandler):
    def __init__(self, socketio):
        self.socketio = socketio
        self.debounce_timer = None
        
    def on_created(self, event):
        if event.src_path.endswith('.pth'):
            self._debounce_emit()
            
    def on_deleted(self, event):
        if event.src_path.endswith('.pth'):
            self._debounce_emit()
            
    def _debounce_emit(self):
        if self.debounce_timer:
            self.debounce_timer.cancel()
        self.debounce_timer = threading.Timer(0.5, self._emit_model_change)
        self.debounce_timer.start()
        
    def _emit_model_change(self):
        self.socketio.emit('model_directory_changed')

def setup_model_watcher(app, model_path):
    socketio = SocketIO(app, cors_allowed_origins="*")
    event_handler = ModelDirectoryHandler(socketio)
    observer = Observer()
    observer.schedule(event_handler, model_path, recursive=False)
    observer.start()
    return socketio