# The ERNIE speed model is used to obtain the basic flow and node extraction is performed.
#  Later switched to the ERNIE 4 Turbo model and stopped using speed.

# 读取.json文件中的uc
def read_uc_in_multi_json(uc_file_list):
    uc_cases_list = []  # 两个公开数据集，都存放在这个列表中，每个公开数据集的数据为一个小列表
    for file in uc_file_list:
        uc_cases_list = read_uc_from_json(file)
    return uc_cases_list

def get_pred_steps_and_write_in_json_eTour(uc_cases_list, Ernie_easyClinic_steps_path):
    out_list = []
    for uc in uc_cases_list:
        use_case = {"index": None, "ucName": None, "pred_steps": None, "dataset": None}
        use_case["index"] = uc["index"]
        use_case["ucName"] = uc["ucName"]
        use_case["dataset"] = uc["dataset"]

        if "uctext" in uc:
            use_case["pred_steps"] = req_analysis_uctext_ucname_easyClinic(uc["uctext"],
                                                                           uc[
                                                                               "ucName"])
        else:
            use_case["pred_steps"] = requirement_analysis_ucname(uc["ucName"])

        for key, value in use_case.items():
            if value is None:
                print(f"错误！key '{key}' 的 value 为 None")

        out_list.append(use_case)
    with open(Ernie_easyClinic_steps_path, 'w', encoding='utf-8') as f:
        for dict_1 in out_list:
            f.write(json.dumps(dict_1, ensure_ascii=False) + '\n')

def req_analysis_uctext_ucname_easyClinic(uctext, ucName):
    # 任务是输入uctext（和uc+path），让LLM输出uc step。
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "You are an expert in software requirement analysis and software design. Please design the functional steps to implement the use case based on the given use case name and use case content description."
                           "Please output the generated functional steps in the following JSON format and do not output any other information："
                           "{"
                           "Functional Steps"':'"[Here is the output.]"
                           "}"
                           "Input: Use case name: Input anagrafica of a laboratory."
                           "Use case content description: It allows the operator to enter the anagrafica of a laboratory analysis or any data that the characterize."
            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of the anagrafica input service Laboratory',"
                           "'View the mask to enter information needed',"
                           "'Inserts data about the anagrafica of laboratory',"
                           "'Confirm placement',"
                           "'Verify the data entered',"
                           "'Stores data',"
                           "'Notify that the operation it is finished with success']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: Delete visit."
                           "Use case content description: It allows the operator to delete a visit previously recorded."
            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service elimination of a visit',"
                           "'View the list of visits in chronological order',"
                           "'Select the visit to delete',"
                           "'Confirm the selection',"
                           "'View the mask for viewing of visit',"
                           "'Confirm the delete',"
                           "'Delete the visit and examinations related to it',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: Input examination."
                           "Use case content description: It allows the operator to record results of a examination supported by a patient required a visit held in outpatient."
            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of input service of data from an examination',"
                           "'View the mask to record of examination',"
                           "'Input necessary data to registration of examination',"
                           "'Confirm input',"
                           "'Verify the data inserted by operator',"
                           "'Stores data confirm the elimination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: Changing examination."
                           "Use case content description: It allows the operator to change a examination previously recorded."
            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service to modify an examination',"
                           "'View a list of tests carried out in chronological order',"
                           "'Select the examination by change',"
                           "'Confirm your selection ',"
                           "'View the mask for the modification of a examination',"
                           "'Change of data ',"
                           "'Confirm Changes ',"
                           "'Verify the data inserted by operator ',"
                           "'updates the examination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: Delete examination."
                           "Use case content description: It allows the operator to delete a Visit previously recorded."
            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service elimination of an examination',"
                           "'View a list of examinations conducted in chronological order',"
                           "'Select the examination to change',"
                           "'Confirm your selection ',"
                           "'View the mask for viewing examination',"
                           "'confirms the deletion',"
                           "'Delete the examination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name:" + ucName +
                           "Use case content description:" + uctext
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)

    if isinstance(result, dict):
        step_list = result.get('result')
        match = re.search(r'\[(.*?)\]', step_list, re.DOTALL)
        if match:
            content = match.group(1)
            step_list = [line.strip() for line in content.strip().split('\n') if line.strip()]
            return step_list
        else:
            match = re.search(r'\{(.*?)\}', step_list, re.DOTALL)
            if match:
                content = match.group(1)
                step_list = [line.strip() for line in content.strip().split('\n') if line.strip()]

                return step_list

    step_list = req_analysis_uctext_ucname_easyClinic(uctext, ucName)
    return step_list

def requirement_analysis_ucname(ucName):
    # 任务是输入uctext（和uc+path），让LLM输出uc step。
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_speed?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "You are an expert in software requirement analysis and software design. Please design the functional steps to implement the use case based on the given use case name."
                           "Please output the generated functional steps in the following JSON format and do not output any other information："
                           "{"
                           "Functional Steps"':'"[Here is the output.]"
                           "}"
                           "Input: Use case name: InputAnagraficaOfLaboratory."

            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of the anagrafica input service Laboratory',"
                           "'View the mask to enter information needed',"
                           "'Inserts data about the anagrafica of laboratory',"
                           "'Confirm placement',"
                           "'Verify the data entered',"
                           "'Stores data',"
                           "'Notify that the operation it is finished with success']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: DeleteVisit."

            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service elimination of a visit',"
                           "'View the list of visits in chronological order',"
                           "'Select the visit to delete',"
                           "'Confirm the selection',"
                           "'View the mask for viewing of visit',"
                           "'Confirm the delete',"
                           "'Delete the visit and examinations related to it',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: InputExamination."

            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of input service of data from an examination',"
                           "'View the mask to record of examination',"
                           "'Input necessary data to registration of examination',"
                           "'Confirm input',"
                           "'Verify the data inserted by operator',"
                           "'Stores data confirm the elimination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: ChangingExamination."

            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service to modify an examination',"
                           "'View a list of tests carried out in chronological order',"
                           "'Select the examination by change',"
                           "'Confirm your selection ',"
                           "'View the mask for the modification of a examination',"
                           "'Change of data ',"
                           "'Confirm Changes ',"
                           "'Verify the data inserted by operator ',"
                           "'updates the examination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name: DeleteExamination."

            },
            {
                "role": "assistant",
                "content": "['The operator activates the execution of service elimination of an examination',"
                           "'View a list of examinations conducted in chronological order',"
                           "'Select the examination to change',"
                           "'Confirm your selection ',"
                           "'View the mask for viewing examination',"
                           "'confirms the deletion',"
                           "'Delete the examination ',"
                           "'Notify the operator that the operation was concluded successfully']"
            },
            {
                "role": "user",
                "content": "Input: Use case name:" + ucName
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    result = json.loads(response.text)

    if isinstance(result, dict):
        step_list = result.get('result')
        match = re.search(r'\[(.*?)\]', step_list, re.DOTALL)
        if match:
            content = match.group(1)
            step_list = [line.strip() for line in content.strip().split('\n') if line.strip()]
            return step_list
        else:
            match = re.search(r'\{(.*?)\}', step_list, re.DOTALL)
            if match:
                content = match.group(1)
                step_list = [line.strip() for line in content.strip().split('\n') if line.strip()]

                return step_list

    step_list = requirement_analysis_ucname(ucName)
    return step_list

def get_pred_node_from_public_dataset(steps_path, out_path):
    # 1、读取uc
    use_case_list = []
    with open(steps_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e}")

    # 2、获取ernie输出的预测node,并写入文件
    with open(out_path, 'w', encoding='utf-8') as f:
        out_list = []
        for uc in use_case_list[188:189]:
            print(f"{uc['index']}")
            out_dict = {"index": None, "ucName": None, "pred_steps": None, "pred_nodes": [], "dataset": None}
            out_dict["dataset"] = uc["dataset"]
            out_dict["index"] = uc["index"]
            out_dict["ucName"] = uc["ucName"]
            out_dict["pred_steps"] = judge_list_is_one_groud_truth(uc["pred_steps"])  # 如果uc["predict_step"]写成了一个项，则拆分
            for pred_step in out_dict["pred_steps"]:
                act = act_extraction(pred_step)
                out_dict["pred_nodes"].append(act)
                print(f"out_dict.append(act):{act}")
                obj = obj_recognition(pred_step)
                print(f"out_dict.append(obj):{obj}")
                out_dict["pred_nodes"].append(obj)
            out_list.append(out_dict)

            for key, value in out_dict.items():
                if value is None:
                    print(f"错误！key '{key}' 的 value 为 None")

            f.write(json.dumps(out_dict, ensure_ascii=False) + '\n')



if __name__ == '__main__':

    # 确认要完成的 task_index
    print("task0-6中很多是操作ernie speed 8k 模型时用的，task7开始改成了 ernie 4 Turbo 8k/ 3.5 ！！ ")
    task_index = 16  # 选择需要完成的task
    print(f"******** start # {task_index} task !!********", datetime.now().strftime("%H:%M:%S"))

    # task_1: 用LLM（Ernie）预测step (prediction steps)
    if task_index == 1:
        print("************* Step 2 ***********", datetime.now().strftime("%H:%M:%S"))
        # 2、读取.json文件中的uc,入参为地址的列表
        uc_file_list = [uc_pub_8in1]
        uc_cases_list = read_uc_in_multi_json(uc_file_list)
        print("************* Step 3 ***********", datetime.now().strftime("%H:%M:%S"))
        # 3、获取每个列表用大模型预测出的steps
        get_pred_steps_and_write_in_json_eTour(uc_cases_list, Ernie_8in1_steps_path)

        # get_pred_steps_and_write_in_json_easyClinic(uc_cases_list, Ernie_easyClinic_steps_path)

        # # iTrust的数据做了一些修改：把steps的None都改成了uctext.
        # get_pred_steps_and_write_in_json_iTrust(uc_cases_list,Ernie_iTrust_steps_path)
        # get_pred_steps_and_write_in_json_keepass(uc_cases_list, Ernie_keepass_steps_path)

    # task_2: 提取pred_step中的node
    elif task_index == 2:
        print("************* Step 4 ***********", datetime.now().strftime("%H:%M:%S"))
        # # 4、读取Ernie的steps预测结果。获取steps中每句话的act、obj
        files_list = [[Ernie_8in1_steps_path, Ernie_8in1_path]]

        for [steps_path, out_path] in files_list:
            get_pred_node_from_public_dataset(steps_path, out_path)