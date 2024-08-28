import os
import os.path as osp

# Directories
project_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
model_dir = osp.join(project_dir, "models")
log_dir = osp.join(project_dir, "logs")
data_dir = osp.join(project_dir, "data")
metadata_dir = osp.join(data_dir, "metadata")
audio_dir = osp.join(data_dir, "audio")
transcript_dir = osp.join(data_dir, "transcript")
CHUNK_DIR = osp.join(data_dir, "chunks")
EMBED_DIR = osp.join(data_dir, "embeddings")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)
os.makedirs(transcript_dir, exist_ok=True)
os.makedirs(CHUNK_DIR, exist_ok=True)
os.makedirs(EMBED_DIR, exist_ok=True)

# Playlist IDs
playlist_id_to_title = {
                "PLDqahtm2vA70VohJ__IobJSOGFJ2SdaRO": "AskSwami Q&A",
                "PLDqahtm2vA728mT-GFH6F-vN2YsS1h72x": "Drg Drsya Viveka",
                "PLDqahtm2vA710Q4PA2yKn8kqS0y-SZNo7": "Aparokshanubhuti",
                "PLDqahtm2vA70ccqIRFR_lipqKvxrHBRRw": "Vedantasara",
                "PLDqahtm2vA72naWj1foEqGFQiN_bRI5my": "Katha Upanishad",
                "PLDqahtm2vA70JIuPYDyWpNBIcRhb7DGCL": "Mundaka Upanishad",
                "PL2imXor63HtRJbtP4mMt-Q2ke8XOkL7pX": "Mandukya Karika",
                "PL2imXor63HtS4ewIKryBL4ZVeiaH8Ij4R": "Bhagavad Gita",
                "PLDqahtm2vA71mxVB2YXXi4J8aYVHXL_bE": "Panchadasi",
                "PLDqahtm2vA72mjPDT6KaPiKdcjlWQC83s": "Upadesa Saram",
                "PLDqahtm2vA70iOJ5q9JOJNvDAgAMkaFj2": "Adhyasa Bhashya",
                "PLDqahtm2vA73W7LK13e9K9Uy1rKcKa1ay": "Vakya Vritti",
                "PLDqahtm2vA71wXmmUKV5T-vuP7v7NY_3t": "Meditation",
                "PLDqahtm2vA70Ojyw-e9DV6K6CPnpu-GIn": "Ashtavakra Samhita",
                "PLDqahtm2vA73NhIqr_6fj4-gF8LqqFo_M": "Yoga Vasistha Sara",
                "PLDqahtm2vA717IaOZp0s9lZqzwmq-MY5u": "The Four Yogas",
                "PLDqahtm2vA72BT1Y7GL1VL7bBEZA7C9vs": "Swami Vivekananda's Jnana Yoga",
                "PLDqahtm2vA738EHCbzJdzOaKcfzyuL51e": "Practical Vedanta",
                "PLDqahtm2vA73pdnNxGLfTQCG4dLbXY6Ja": "Advaita Vedanta",
                "PLDqahtm2vA729T2LPOg9FPelznaM-vTIk": "Mandukya Upanishad",
                "PLDqahtm2vA72vvR5GYAlkuJylJC5_kGXA": "The Lamp of Bliss",
                "PLDqahtm2vA70flIbPCi4vqZ33gFda-THY": "Other Venues",
                "PLDqahtm2vA72ilWvaXhKRDUemEsz4VCKd": "Lectures on Vedanta",
                }
