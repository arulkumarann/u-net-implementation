import os
import subprocess
import zipfile

def download_carvana_dataset(data_folder="data/raw/"):

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    competition = "carvana-image-masking-challenge"
    
    command = f"kaggle competitions download -c {competition} -p {data_folder}"
    
    try:
        print("Downloading Carvana dataset...")
        subprocess.run(command, shell=True, check=True)
        print("Download complete!")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading the dataset: {str(e)}")
        return

def extract_zip_files(folder = r'/teamspace/studios/this_studio/u_net_implementation/data/raw'):
 
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".zip"):
                zip_path = os.path.join(root, file)
                extract_folder = os.path.splitext(zip_path)[0]  # Create a folder with the same name as the zip file
                
                print(f"Extracting: {zip_path}")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_folder)
                
                print(f"Extracted to: {extract_folder}")

if __name__ == "__main__":
    download_carvana_dataset()
    extract_zip_files()
