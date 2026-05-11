import zipfile
from pathlib import Path
import random

# Usa rutas absolutas para evitar confusiones de dónde se ejecuta el script
zip_path = Path(r"D:\VC\PIA\630DS\Images_Courriers.zip")
output_dir = Path(r"D:\VC\PIA\630DS\R630L")
output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    all_files = z.namelist()
    
    # DEBUG: Imprime los primeros 5 archivos para ver la ruta real interna
    # print("Ejemplo de rutas dentro del zip:", all_files[:5])

    # Filtramos de forma más flexible: que contenga 'DVD1_TIF' y termine en L.tif o L.xml
    # Usamos "in f" por si la carpeta raíz tiene un nombre ligeramente distinto
    dvd1_list = [f for f in all_files if "DVD1_TIF" in f and f.lower().endswith(('l.tif', 'l.xml'))]
    dvd2_list = [f for f in all_files if "DVD2_TIF" in f and f.lower().endswith(('l.tif', 'l.xml'))]
    dvd3_list = [f for f in all_files if "DVD3_TIF" in f and f.lower().endswith(('l.tif', 'l.xml'))]

    def get_random_samples(file_list, count):
        if not file_list:
            return []
        # Agrupamos por el nombre del archivo sin extensión
        unique_letters = list(set([Path(f).stem for f in file_list]))
        samples = random.sample(unique_letters, min(count, len(unique_letters)))
        
        # Buscamos los archivos originales que coincidan con los nombres elegidos
        return [f for f in file_list if Path(f).stem in samples]

    all_to_extract = (get_random_samples(dvd1_list, 210) + 
                      get_random_samples(dvd2_list, 210) + 
                      get_random_samples(dvd3_list, 210))

    print(f"Archivos encontrados para extraer: {len(all_to_extract)}")
    
    for file in all_to_extract:
        # Esto extraerá manteniendo la estructura de carpetas dentro de R630L
        z.extract(file, output_dir)
        
print(f"¡Listo! Revisa la carpeta: {output_dir}")