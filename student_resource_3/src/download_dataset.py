import pandas as pd
from src.utils import download_images

def main():
    # Load the train.csv file
    train_data = pd.read_csv('dataset/train.csv')

    # Extract the image links from the CSV
    image_links = train_data['image_link'].tolist()

    # Specify the folder where the images will be downloaded
    download_folder = 'downloaded_images/'

    # Download all images using multiprocessing for speed
    download_images(image_links, download_folder, allow_multiprocessing=True)

if __name__ == '__main__':
    main()
