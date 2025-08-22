from PIL import Image
import os
from instaloader import Instaloader, Profile
import lzma
import json
import shutil
from datetime import datetime, timedelta

def sort_files_by_extension(base_path):
    os.makedirs(os.path.join(base_path, "xz"), exist_ok=True)
    file_types = ['jpg', 'xz']
    for file_type in file_types:
        type_folder = os.path.join(base_path, file_type)

        for filename in os.listdir(os.path.join(base_path, "raw")):
            if filename.endswith(f".{'png'}"):
                print(f"Bilddatei im Format .png erkannt.")
                png_image = Image.open(os.path.join(os.path.join(base_path, "raw"), filename))
                png_image.convert('RGB').save(os.path.join(os.path.join(base_path, "raw"), filename), 'JPEG')
                print(f"{filename} verschieben nach {os.path.join(type_folder, filename[:-4] + ".jpg")}")
                os.rename(
                    os.path.join(os.path.join(base_path, "raw"), filename),
                    os.path.join(type_folder, filename[:-4] + ".jpg")
                )
            elif filename.endswith(f".{file_type}"):
                print(f"{filename} verschieben nach {os.path.join(type_folder, filename)}")
                os.rename(
                    os.path.join(os.path.join(base_path, "raw"), filename),
                    os.path.join(type_folder, filename)
                )

    shutil.rmtree(os.path.join(base_path, "raw"))

# .xz Dateien zu entpacken
def extract_xz_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".xz"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename[:-3])  # ".xz" entfernen

            try:
                # Datei entpacken
                with lzma.open(input_path, "rt", encoding="utf-8") as compressed_file:
                    json_content = compressed_file.read()

                try:
                    json.loads(json_content)
                except json.JSONDecodeError:
                    print(f"Warnung: {filename} enthält keine gültige JSON-Datei.")
                    continue

                # Entpackten Inhalt speichern
                with open(output_path, "w", encoding="utf-8") as json_file:
                    json_file.write(json_content)

                print(f"{filename} erfolgreich entpackt nach {output_path}")

            except Exception as e:
                print(f"Fehler beim Verarbeiten von {filename}: {e}")
    shutil.rmtree(input_folder)

# letzte Bilddatum im Profil zu finden
def get_last_image_datetime(profile_folder):
    jpg_folder = os.path.join(profile_folder, "jpg")
    last_datetime = None

    # das neueste Bild basierend auf Datum und Zeit im Dateinamen finden
    for filename in os.listdir(jpg_folder):
        if filename.endswith(".jpg"):
            # Dateiname im Format JJJJ-MM-TT_HH-MM-SS.jpg erwartet
            try:
                file_datetime = datetime.strptime(filename[:19], "%Y-%m-%d_%H-%M-%S")
                if last_datetime is None or file_datetime > last_datetime:
                    last_datetime = file_datetime
            except ValueError:
                continue

    return last_datetime

# Hauptfunktion
def download_and_sort_instagram_data(profile_name, base_download_path, start_date, end_date, update_metadata, start_date_metadata):
    # Überprüfen ob Ordner des Profils bereits existiert
    profile_folder = os.path.join(base_download_path, profile_name)
    
    # Wenn der Ordner bereits existiert die letzten Post-Daten anpassen
    if os.path.exists(profile_folder):
        print(f"Download für das Profil {profile_name} wurde bereits durchgeführt.")
        
        # Datum des letzten Bildes aus dem JPG-Ordner extrahieren
        last_image_datetime = get_last_image_datetime(profile_folder)
        if last_image_datetime:
            print(f"Das letzte bereits heruntergeladene Bild wurde am {last_image_datetime.strftime('%Y-%m-%d %H:%M:%S')} hochgeladen.")
            
            # Wenn das letzte Bild vor dem Enddatum liegt, den Zeitraum anpassen
            if last_image_datetime < end_date:
                # Das neue Startdatum ist der Tag nach dem letzten Bild
                start_date = last_image_datetime + timedelta(seconds=1)
                print(f"Der Zeitraum wird angepasst: Neues Startdatum ist {start_date.strftime('%Y-%m-%d %H:%M:%S')}.")
                if update_metadata and start_date_metadata > start_date:
                    start_date_metadata = start_date
            else:
                print(f"Das Enddatum {end_date.strftime('%Y-%m-%d %H:%M:%S')} liegt vor dem letzten bereits heruntergeladenen Post vom {last_image_datetime.strftime('%Y-%m-%d %H:%M:%S')}. Der Download wird abgebrochen.")
                return
        else:
            print("Keine Bilddateien im Ordner gefunden.")
        
        print("Skript wird fortgesetzt.")
    else:
        print(f"Profilordner für {profile_name} existiert noch nicht. Es wird eine Ordnerstruktur angelegt und der Download gestartet.")
        
        # Ordnerstruktur für den ersten Download erstellen
        os.makedirs(profile_folder, exist_ok=True)
        os.makedirs(os.path.join(profile_folder, "jpg"), exist_ok=True)
        os.makedirs(os.path.join(profile_folder, "json"), exist_ok=True)

    # Instaloader initialisieren
    all_loader = Instaloader( 
        download_pictures=True,
        download_videos=False, 
        download_video_thumbnails=False,
        compress_json=True, 
        download_geotags=False, 
        post_metadata_txt_pattern=None, 
        max_connection_attempts=0,
        download_comments=False,
        )
    
    metadata_loader = Instaloader( 
        download_pictures=False,
        download_videos=False, 
        download_video_thumbnails=False,
        compress_json=True, 
        download_geotags=False, 
        post_metadata_txt_pattern=None, 
        max_connection_attempts=0,
        download_comments=False,
        )

    # Profil laden
    profile = Profile.from_username(all_loader.context, profile_name)
    posts = profile.get_posts()

    metadata_profile = Profile.from_username(metadata_loader.context, profile_name)
    metadata = metadata_profile.get_posts()

    # Arbeitsverzeichnis ändern, damit Instaloader die Dateien im richtigen Ordner speichert
    original_cwd = os.getcwd()
    os.chdir(profile_folder)

    downloaded_posts = 0  

    # Nur Bilder herunterladen, die im angegebenen Zeitraum liegen
    for post in posts:
        post_datetime = post.date_utc
        if start_date <= post_datetime <= end_date and not post.is_video:
            all_loader.download_post(post, target="raw")
            downloaded_posts += 1
        elif post_datetime < start_date:
            break

    #Metadaten updaten
    if update_metadata:
        shutil.rmtree(os.path.join(profile_folder, "json"))
        os.makedirs(os.path.join(profile_folder, "json"), exist_ok=True)
        for data in metadata:
            metadata_datetime = data.date_utc
            if start_date_metadata <= metadata_datetime < start_date:
                print(metadata_datetime)
                metadata_loader.download_post(data, target="raw")
                downloaded_posts += 1
            elif metadata_datetime < start_date_metadata:
                break                

    os.chdir(original_cwd)

    # Wenn keine Bilder heruntergeladen wurden, Skript beenden
    if downloaded_posts == 0:
        print(f"Keine Bilder im angegebenen Zeitraum ({start_date.strftime('%Y-%m-%d %H:%M:%S')} bis {end_date.strftime('%Y-%m-%d %H:%M:%S')}) gefunden.")
        return

    # Dateien nach Typ sortieren
    sort_files_by_extension(profile_folder)

    # .xz Dateien entpacken
    xz_folder = os.path.join(profile_folder, "xz")
    json_folder = os.path.join(profile_folder, "json")
    extract_xz_files(xz_folder, json_folder)

if __name__ == "__main__":
    PROFILE = "csu"
    BASE_DOWNLOAD_PATH = r"C:\Users\lukaspasold\databaseimages"
    
    start_date = datetime(2001, 12, 18)
    end_date = datetime(2024, 12, 18)

    #Metadaten können auch weiter in die Vergangenheit aktualisiert werden ohne die jpg neu herunterzuladen
    update_metadata = False
    start_date_metadata = datetime(2000, 1, 1)
    
    download_and_sort_instagram_data(PROFILE, BASE_DOWNLOAD_PATH, start_date, end_date, update_metadata, start_date_metadata)