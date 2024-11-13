# programm to change a csv-file into a json-file 
import pandas as pd

# reading the csv-file and turning it into a data frame 
csv_file = '/Users/chanti/Desktop/5. Semester/Softwareprojekt/Code/Emotion-Detection/BitteLabeln.csv'  # Pfad zur CSV-Datei
df = pd.read_csv(csv_file, encoding='utf-8')

# Konvertiere das DataFrame in JSON und speichere es
json_file = 'BitteLabeln.json'  # Pfad zur JSON-Ausgabedatei
df.to_json(json_file, orient='records', force_ascii= False ,lines=False)  # Für eine list-basierte JSON-Datei

print(f"CSV wurde erfolgreich in {json_file} umgewandelt!")