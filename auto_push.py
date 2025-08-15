import time
import datetime
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class GitAutoPushHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        print("Changes detected... pushing to GitHub.")
        message = (
            f"Auto update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        os.system("git add .")
        os.system(f'git commit -m "{message}"')
        os.system("git push origin main")


if __name__ == "__main__":
    path = "."  # Watch current directory
    event_handler = GitAutoPushHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print("Watching for changes... Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
