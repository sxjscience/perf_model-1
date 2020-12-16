import os
import glob
import json


def parse_folder_results(folder):
    out_json_path_name = f'{folder}_analysis.json'
    paths = sorted(glob.glob(f'{folder}/*_best_op_results.json'))
    assert len(paths) == 11, paths

    avg_result_dic = dict()


    def update_result(avg_result, file_path):
        with open(file_path, 'r') as in_f:
            dat = json.load(in_f)
        for key, idx, thrpt in dat:
            name = key[0]
            if name in avg_result:
                avg_result[name]['thrpt'].append(thrpt)
                avg_result[name]['rank'].append(idx)
            else:
                avg_result[name] = dict()
                avg_result[name]['thrpt'] = [thrpt]
                avg_result[name]['rank'] = [idx]


    for path in paths:
        update_result(avg_result_dic, path)

    for name in avg_result_dic:
        num = len(avg_result_dic[name]['thrpt'])
        avg_result_dic[name]['avg_thrpt'] = sum(avg_result_dic[name]['thrpt']) / num
        avg_result_dic[name]['avg_rank'] = sum(avg_result_dic[name]['rank']) / num
        print(f'{name}, {avg_result_dic[name]["avg_thrpt"]},'
              f' {avg_result_dic[name]["avg_rank"]}')
    with open(out_json_path_name, 'w') as out_f:
        json.dump(avg_result_dic, out_f)


for base_folder in ['cat_regression_op_5000_split1_e2e_g4_npara8_ntop8_seed123',
                    'nn_regression_op_new_split1_-1_1000_512_3_0.1_0_e2e_g4_npara8_ntop8_seed123',
                    'nn_regression_op_new_split1_-1_1000_512_3_0.1_1_e2e_g4_npara8_ntop8_seed123',
                    'nn_regression_op_new_split0.7_-1_1000_512_3_0.1_1_e2e_g4_npara8_ntop8_seed123',
                    'nn_regression_op_new_split0.5_-1_1000_512_3_0.1_1_e2e_g4_npara8_ntop8_seed123',
                    'nn_regression_op_new_split0.3_-1_1000_512_3_0.1_1_e2e_g4_npara8_ntop8_seed123']:
    print('Folder=', base_folder)
    parse_folder_results(base_folder)
