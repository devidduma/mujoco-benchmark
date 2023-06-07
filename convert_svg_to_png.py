import os
import pyvips

def batch_svg_files(path, choice="list"):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.svg'):
                if choice=="convert":
                    convert_to_png(os.sep.join([dirpath, filename]))
                elif choice=="list":
                    print(os.sep.join([dirpath, filename]))

def convert_to_png(relative_path):
    image = pyvips.Image.new_from_file(relative_path, dpi=300)
    image.write_to_file(relative_path[:-4]+".png")

if __name__ == '__main__':
    print(batch_svg_files(path="./benchmark/plots", choice="convert"))
