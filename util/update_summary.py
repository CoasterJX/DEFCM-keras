import csv
import os


def generate_summary_results(test_path, optimum_standard='facc', optimizer=max):

    headers = [
        'group',
        'iacc', 'inmi', 'iari', 'iloss', 'itime',
        'facc', 'fnmi', 'fari', 'floss', 'ftime']

    summary_file = open(os.path.join(test_path, 'result-summary.csv'), 'w')
    summary_writer = csv.DictWriter(summary_file, fieldnames=headers)
    summary_writer.writeheader()

    for trial_folder in sorted(os.listdir(test_path)):

        summary_row = [trial_folder]
        trial_path = os.path.join(test_path, trial_folder)
        if not os.path.isdir(trial_path):
            continue
        if 'metrics-per-iter.csv' in os.listdir(trial_path):

            with open(os.path.join(trial_path, 'metrics-per-iter.csv'), 'r') as f:
                reader = csv.DictReader(f)
                metrics = [[row['acc'], row['nmi'], row['ari'], row['loss'], row['time']] for row in reader]

                summary_row.extend(metrics[0])
                summary_row.extend(metrics[-1])
        
        else:
            generate_summary_results(trial_path)

            with open(os.path.join(trial_path, 'result-summary.csv'), 'r') as f:
                reader = csv.DictReader(f)
                res_list = [round(float(row[optimum_standard]), 5) for row in reader]
                selected_ind = res_list.index(optimizer(res_list))
            
            with open(os.path.join(trial_path, 'result-summary.csv'), 'r') as f:
                reader = csv.DictReader(f)
                i = 0
                for row in reader:
                    if i == selected_ind:
                        metrics = []
                        for h in headers[1:]:
                            metrics.append(row[h])
                        summary_row.extend(metrics)
                        break
                    i += 1
        
        sr = {}
        for i in range(len(headers)):
            sr[headers[i]] = summary_row[i]
        summary_writer.writerow(sr)

    summary_file.close()
