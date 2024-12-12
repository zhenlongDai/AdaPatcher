from codeTool.utlis.utils import load_list_from_json
import json
import argparse

def count_matching_scores(file_path, pattern, ChooseSID_path, second_eval_pattern, length):
    data = load_list_from_json(file_path)
    count = 0
    TotalCount = 0
    if pattern == "test":
        SIDSet = getDcit(ChooseSID_path)
    
    if "retention_rate" in data[0]:
        RR_TAG = True
    else:
        RR_TAG = False
    AC_retention_rate=0
    Total_retention_rate=0
    AC_overall_rate=0
    Total_overall_rate=0

    same_count_in_AC = 0
    same_count = 0

    for item in data:
        SId = item['submission1_id']
        if pattern == "test" and SId not in SIDSet: continue
       


        if pattern == "dev":
            if second_eval_pattern == "dev_1":
                if "FL" in SId: continue
            elif second_eval_pattern == "dev_2":
                if "FL" not in SId: continue
                SId = SId[:-3]
            else:
                if "FL" in SId:
                    SId = SId[:-3]
        
        if "flag" in item and item['flag'] == False: 
            if RR_TAG == True:
                    if item['added_lines'] != 0 or item['removed_lines'] != 0:
                        Total_retention_rate += item['retention_rate']
                        a = item['added_lines']
                        b = item['removed_lines']
                        s = item['code1_lines']
                        Total_overall_rate +=  (s-b)*1.0/(s+a-b)       
            TotalCount += 1
            continue
        
        if item['code1_lines'] < length:
            continue

        if RR_TAG == True:
            if item['added_lines'] == 0 and item['removed_lines'] == 0:
                same_count += 1
            else:
                Total_retention_rate += item['retention_rate']
                a = item['added_lines']
                b = item['removed_lines']
                s = item['code1_lines']
                Total_overall_rate +=  (s-b)*1.0/(s+a-b)  

        if 'code_test_score' in item and 'TotalScore' in item:
            if item['code_test_score'] == item['TotalScore'] and item['TotalScore'] != 0:
                count+=1
                if RR_TAG == True:
                    if item['added_lines'] == 0 and item['removed_lines'] == 0:
                        same_count_in_AC += 1
                    else:
                        AC_retention_rate += item['retention_rate']
                        a = item['added_lines']
                        b = item['removed_lines']
                        s = item['code1_lines']
                        AC_overall_rate +=  (s-b)*1.0/(s+a-b)

                

        TotalCount += 1
    if RR_TAG == True:
        avg_retention_rate = Total_retention_rate/TotalCount
        avg_overall_rate =  Total_overall_rate/TotalCount
        
        avg_AC_retention_rate = AC_retention_rate/count
        avg_AC_overall_rate =  AC_overall_rate/count
        print("#retention_rate-------")
        print(f"TotalCount = {TotalCount}")
        print(f"Total_overall_rate = {Total_overall_rate}")
        print(f"avg_overall_rate = {avg_overall_rate}")
        # print(f"avg_AC_overall_rate = {avg_AC_overall_rate}")
        # print(f'avg_retention_rate = {avg_retention_rate}')
        # print(f'avg_AC_retention_rate = {avg_AC_retention_rate}')
        print(f"same_count= {same_count}")
        #print(f"same_count_in_AC= {same_count_in_AC}")
        print("-------")

    print(f'TotalCount = {TotalCount}')
    rate = count*1.0 /TotalCount
    print(f'ACC rate = {rate}')
    return count

def cal_rate(baseResultList, newResultList, TotalScore, base_test_score):
    flag = True
    if len(newResultList) == 0: #Compilation error
        if base_test_score!=0:
            flag = False 
        return flag, 0.0, 0
    # Use list comprehensions to turn elements less than 0 into 0
    newResultList = [x if x >= 0 else 0 for x in newResultList]
    
    
    if len(baseResultList) != len(newResultList):
        raise ValueError("two list have diffrent length")
    count = 0
    for a, b in zip(baseResultList, newResultList):
        if a == 1 and b == 0:
            flag = False
            break
        if a == 0 and b == 1:
            count += 1
    if flag == False:
        count = 0
    #if count == 0:
    #    flag = False

    rate = count*1.0/(TotalScore-base_test_score)
    return flag, rate, count

def getDcit(ChoseSID_path):
    base_list = load_list_from_json(ChoseSID_path)
    tmpSet = set()
    for item in base_list:
        Id = item['submission1_id']
        tmpSet.add(Id)
    print(f"Set num = {len(tmpSet)}")
    return tmpSet

def cal_improve_rate(data_path, file_path, pattern, ChooseSID_path = None, second_eval_pattern = None):
    base_path = data_path + f'{pattern}.json'
    base_list = load_list_from_json(base_path)
    file_list = load_list_from_json(file_path)
    submission2resultMap = dict()
    if pattern == "test":
        SIDSet = getDcit(ChooseSID_path)
    for item in base_list:
        Id = item['submission1_id']
        code_test_status = item['code1_test_status']
        base_test_score = item['code1_test_score']
        submission2resultMap[Id] = (code_test_status,base_test_score)
    total_improve_rate = 0.0
    ErrorCount = 0
    recordCount = 0
    IRCount = 0
    Improve_retention_rate=0
    Improve_overall_rate=0
    if "retention_rate" in file_list[0]:
        RR_TAG = True
    else:
        RR_TAG = False

    for item in file_list:
        SId = item['submission1_id']
        #print(SId)
        if pattern == "test" and SId not in SIDSet: continue
        if pattern == "dev":
            if second_eval_pattern == "dev_1":
                if "FL" in SId: continue
            elif second_eval_pattern == "dev_2":
                if "FL" not in SId: continue
                SId = SId[:-3]

            else:
                if "FL" in SId:
                    SId = SId[:-3]

        newResultList = item['code_test_status']
        baseResultList,base_test_score = submission2resultMap[SId]
        TotalScore = item['TotalScore']
        if "flag" in item and item['flag'] == False: 
            flag = False
            tmp_rate = 0.0
        else:
            flag,tmp_rate, impove_Testcount = cal_rate(baseResultList, newResultList, TotalScore, base_test_score)
        total_improve_rate += tmp_rate
        
        if flag == False:
            ErrorCount += 1
        else:
            
            if RR_TAG == True and impove_Testcount != 0:
                Improve_retention_rate+= item['retention_rate']
                a = item['added_lines']
                b = item['removed_lines']
                s = item['code1_lines']
                Improve_overall_rate +=  (s-b)*1.0/(s+a-b)  
                IRCount += 1

        recordCount += 1
    
    avg_improve_rate = total_improve_rate/recordCount

    ErrorRate = ErrorCount*1.0/recordCount
    if RR_TAG == True:
        avg_Improve_retention_rate = Improve_retention_rate*1.0/(IRCount)
        avg_Improve_overall_rate = Improve_overall_rate*1.0/(IRCount)
        print("\n##Improve_retention_rate-----------")
        print(f">>>avg_Improve_overall_rate = {avg_Improve_overall_rate}")
        print(f">>>avg_Improve_retention_rate = {avg_Improve_retention_rate}")
        print("-----------")

    print(f">>>recordCount = {recordCount}")
    print(f">>>avg_improve_rate = {avg_improve_rate}")
    print(f">>>ErrorCount = {ErrorCount}")
    print(f">>>ErrorRate = {ErrorRate}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Statistical Script")
    parser.add_argument('--data_path', type=str, default='./repairDataset/RepairData-PythonLevel/normalDataset-Full/', required=False, help='eval data path')
    parser.add_argument('--file_path', type=str, default='./predict_evalResult_dir/Exec_checkpoint-7400_test.json', required=True, help='Input data for evaluation')
    parser.add_argument('--eval_pattern', type=str, default='dev', required=True, help='eval_pattern:test/dev')
    parser.add_argument('--second_eval_pattern', type=str, default='dev_1', required=True, help='dev_1:origin dev_1:predict dev_1:mix')
    parser.add_argument('--ChooseSID_path', type=str, default='./predict_dir/baseline/trace_baseline_result.json', required=False, help='eval base ChooseSID') 
    parser.add_argument('--ChooseLength', type=int, default='10', required=False, help='code lines large than it') 
    args = parser.parse_args()

    matching_count = count_matching_scores(args.file_path, args.eval_pattern, args.ChooseSID_path, args.second_eval_pattern, args.ChooseLength)
    print(f'Number of elements with code_test_score == TotalScore and TotalScore != 0 is {matching_count}')
    cal_improve_rate(data_path = args.data_path, file_path = args.file_path, pattern= args.eval_pattern,ChooseSID_path = args.ChooseSID_path, second_eval_pattern = args.second_eval_pattern)
