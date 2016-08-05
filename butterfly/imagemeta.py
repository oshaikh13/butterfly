import re
import argparse
import cv2
import glob
import os

class ImageMetaHandler :
 
    # All this does right now is get 'z' slices for any dataset.
    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in re.split(''' (?=(?:[^'"]|'[^']*'|"[^"]*")*$)''', arg_line):
            if not arg.strip():
                continue
            yield arg.strip('\'\"\n')

    @staticmethod
    def parseNumRange(num_arg):
        match = re.match(r'(\d+)(?:-(\d+))?$', num_arg)
        if not match:
            raise argparse.ArgumentTypeError(
                "'" + num_arg + "' must be a number or range (ex. '5' or '0-10').")
        start = match.group(1)
        end = match.group(2) or match.group(1)
        step = 1
        if end < start:
            step = -1
        return list(range(int(start), int(end) + step, step))

    @staticmethod
    def firstSliceInfo(method, base_path) :
        # Assume every "slice" is the same dimension.
        if method == "mojo": 
            tmp_file = os.path.join(
                base_path,
                'w=00000000',
                'z=00000000')

            files = os.listdir(tmp_file);

            currentFile = files[-1];
            delims = currentFile.replace("=", ' ').replace(",", ' ').replace('.', ' ').split(' ');
            
            coordinates = {
                'x' : int(delims[1])+1, 
                'y' : int(delims[3])+1
            };

            default_img_path = os.path.join(
                tmp_file,
                'y=' + delims[3] + ',x=' + delims[1] + '.*');

            img_ext = os.path.splitext(glob.glob(default_img_path)[0])[1]
            default_img_path = default_img_path.replace('.*', img_ext);

            img = cv2.imread(default_img_path);

            coordinates['x'] = coordinates['x'] * img.shape[0];
            coordinates['y'] = coordinates['y'] * img.shape[1];

            return coordinates;

    @staticmethod
    def getDatasourceInformation(datapath) :
        sourceInformation = {
            'depth': -1,
            'height': -1,
            'width': -1,
            'mip': -1
        }

        # A. Try mojo
        if os.path.split(datapath)[-1] == "mojo":
            # Proceed to do mojo-specific hunting.
            base_path = os.path.join(datapath, 'images', 'tiles');
            slice_folders = glob.glob(os.path.join(base_path, 'w=00000000', 'z=*'));
            sourceInformation['depth'] = len(slice_folders);
            size = ImageMetaHandler.firstSliceInfo("mojo", base_path);
            sourceInformation['width'] = size['x'];
            sourceInformation['height'] = size['y'];
        # B. Try reading .args files
        else:
            # Anything not mojo, that has a .args file
            args_file = os.path.join(datapath, '*.args')
            args_file = glob.glob(args_file)[0]

            with open(args_file, 'r') as f:
                args_list = [
                    arg for line in f for arg in ImageMetaHandler.convert_arg_line_to_args(line)]

            zRangeData = ImageMetaHandler.parseNumRange(args_list[args_list.index('--z_ind') + 1]);
            sourceInformation['depth'] = zRangeData[-1] + 1; #include slice one left out by parser


        return sourceInformation;
