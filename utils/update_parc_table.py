import csv
import json
from args import Args
from datetime import datetime
import numpy as np

def get_time_diff(date1, date2):
    # returns date1 - date2
    diff = date1 - date2
    return int(diff.days*100/365)/100


if __name__ == '__main__':
    # First read the json file we want to update
    with open(Args.parc_file) as json_file:
        json_obj = json.load(json_file)

    # Open data file
    with open(Args.data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        header_dict = {}
        sub_to_date = {}
        scan_to_age = {}

        h = ['EXAMDATE', 'AGE']
        for row in csv_reader:
            if line_count == 0:
                for j, header in enumerate(row):
                    header_dict[header] = j
                line_count += 1
            else:
                subid = row[header_dict['PTID']]
                base_age = row[header_dict['AGE']]
                date = datetime.strptime(row[header_dict['EXAMDATE']], '%Y-%m-%d')
                scanid = row[header_dict['subject']]

                if subid not in sub_to_date.keys():
                    # viscode = row[header_dict['VISCODE']]
                    # if not viscode == 'bl':
                    #     print("found bad stuff {}".format(subid))
                    sub_to_date[subid] = date
                    scan_to_age[scanid] = (subid, float(base_age), date)
                else:
                    age = float(base_age) + get_time_diff(date, sub_to_date[subid])
                    scan_to_age[scanid] = (subid, age, date)
                line_count += 1
        print(f'Processed {line_count} lines.')

        age_list = []
        # Update parc table
        for scan, val in scan_to_age.items():
            subid, age, date = val
            age_list.append(age)
            if subid in json_obj.keys():
                for sub_t in json_obj[subid]:
                    if sub_t['network_id'] == scan:
                        sub_t['age'] = age
                        sub_t['date'] = date.strftime('%Y-%m-%d')

        # with open(Args.parc_file.split('.')[0] + "_new.json", 'w') as json_file:
        #     json.dump(json_obj, json_file)

        # Additionally compute mean and std age
        age_list = np.array(age_list)
        print("Age = {:0.2f} +- {:0.2f}".format(age_list.mean(), age_list.std()))