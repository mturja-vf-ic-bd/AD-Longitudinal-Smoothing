import csv
import os
from shutil import copyfile
import json

if __name__ == '__main__':
    with open('/home/turja/AD-Data_mapping.csv', 'r') as map:
        csv_reader = csv.reader(map, delimiter=',')
        line_count = 0
        base_dir = '/home/turja'
        id_to_scan_map = dict()
        for row in csv_reader:
            if line_count > 0:
                if not row[1] in id_to_scan_map.keys():
                    id_to_scan_map[row[1]] = row[0]
                    source = base_dir + '/Desktop/sub/' + row[0] + '/parcellationTable.json'
                    destination = base_dir + '/AD-Data_Organized/' + row[1] + '/helper_files/'
                    destination_smooth = base_dir + '/AD-Data_Organized/' + row[1] + '_smoothed/helper_files/'
                    if not os.path.exists(destination):
                        os.makedirs(destination)
                    if not os.path.exists(destination_smooth):
                        os.makedirs(destination_smooth)

                    if os.path.exists(source):
                        ptable = json.load(open(source, 'r'))
                        del ptable[118]
                        del ptable[76]
                        del ptable[42]
                        del ptable[0]
                        json.dump(ptable, open(os.path.join(destination, 'parcellationTable.json'), 'w'))
                        json.dump(ptable, open(os.path.join(destination_smooth, 'parcellationTable.json'), 'w'))
                    else:
                        print(row[1], 'not found')

            line_count += 1

        print(f'Processed {line_count} lines.')
        print('total keys: ', len(id_to_scan_map.keys()))
