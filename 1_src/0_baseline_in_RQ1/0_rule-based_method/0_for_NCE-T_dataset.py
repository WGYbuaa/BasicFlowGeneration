# class of use case
class useCase:
    def __init__(self):
        self.index_global = 0  # The index of this uc in the global uc.
        self.step_former = []  # The original steps in this use case
        self.step_list = []  # List of steps in this use case.
        self.tc_text = ""   # Original description text.
        self.tc_path = ''  # use case path
        self.tc_embedding = [] # Used to store the embedded tc_text and tc_path splices.
        self.keyword = []  # Used to store the keywords of tc, and the embedding corresponding to each keyword, using a dictionary to store {keyword:embedding}

    def add_ts(self, use_case_step, step):
        self.step_former.append(step)
        self.step_list.append(use_case_step)

    def __repr__(self):
        s = str(self.index_global)
        return s

    def __str__(self):
        return self.__repr__()

# extract action and object from use case step with tool and user domain dictionary.
def extract_action_object(testStep):
    action_word = ''  # Stores an action in a step
    object_word = ''  # Stores an object in a step
    object_lable = {"全局开关": "全局开关", "控制器": "控制器", '的业务': '业务', '查询业务': "业务", '业务删除': "业务",
                    "业务去激活": "业务", "业务激活": "业务", "清除环境": "环境", "校验北向查询结果": "北向查询结果",
                    "在源宿端口上下发业务级别": '业务', "插告警": '告警', "告警上报": '告警', "告警解除": '告警', '业务下发成功': '业务',
                    "时延告警": '告警', "去激活业务": "业务", '主备路径时延发生倒换': '时延值', "的一条路径下发业务": '业务',
                    '业务创建成功': '业务', "告警清除": '告警', '清除故障': '故障', "清除告警": '告警', "网元上配置保护链路": "链路",
                    '激活业务成功': '业务', '同时隙': '时隙', '主备链路': '链路', '使用的': '链路', '保护的业务': '业务', }
    action_lable = {"主备倒换": "主备倒换", "查询业务": '查询', "业务去激活": "去激活", "业务激活": "激活",
                    "校验北向查询结果": "校验", "清除环境": "清除", "在源宿端口上下发业务级别": '下发', '链路取消告警': '取消',
                    "插告警": '插', '清除故障': '清除', '使用的': '使用'}

    # Add user domain dictionary
    test_model.tokenizer.pkuseg_update_user_dict(add_noun_dict_spacy(path='../data/extract_data/properNounDict.json'))

    # Remove special symbols
    for i in string.punctuation:
        testStep = testStep.replace(i, '')
    testStep = re.sub('[\d]', '', testStep)  # Remove numbers
    testStep = re.sub('[\s]', '', testStep)  # Remove spaces
    # testStep = re.sub('[a-zA-Z]', '', testStep)  # Remove English characters
    doc = test_model(testStep)
    for token in doc:
        if token.dep_ == 'ROOT':
            action_word = str(token)
            for token1 in doc:
                if token1.dep_ == 'dobj':
                    object_word = str(token1)
    action_object = [action_word, object_word]

    if len(doc) == 0:
        print("len(doc) == 0", testStep)
    # If action_word is "获取" and object_word is empty, take the last word of the sentence as object_word
    if '获取' in str(doc):
        action_object = ['获取', str(doc[-1])]
    elif action_object[0] == '设置' and action_object_dict['设置'] in str(doc):
        action_object = [action_word, action_object_dict['设置']]
    elif str(doc[0]) == "修改" and action_object_dict['修改'] in str(doc):
        action_object = ["修改", action_object_dict['修改']]

    for obj_lab in object_lable.keys():
        if obj_lab in str(doc):
            action_object[1] = object_lable[obj_lab]
            continue
    for act_lab in action_lable.keys():
        if act_lab in str(doc):
            action_object[0] = action_lable[act_lab]
            continue

    return action_object

def extract_use_case_step_1(path):
    use_case_list = list()  # Store all use cases in order uc.
    use_case = useCase()
    uc_number = 0  # Global index of uc.
    step_number = 0  # Position (index) of step within uc.

    path = path.replace('\\', '/')
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()  # Remove possible spaces.
            if len(line) == 0:  # An extra space is required at the end of the dataset for the last uc to be returned.
                if len(use_case.step_former) > 0:
                    use_case.index_global = uc_number
                    use_case.tc_text = tc_text
                    use_case.tc_path = tc_path
                    use_case_list.append(use_case)  # The sum of the use cases is stored according to one UC
                    use_case = useCase()
                    uc_number = uc_number + 1  # Calculate the index for the next use case.
                    step_number = 0

                    # if len(use_case_list) == 10:  # Use the first thirty use cases to make a demo, and comment out these two lines when you don't need them.
                    #     return use_case_list

                continue

            json_data = json.loads(line)  # in rows, into json_data.

            if 'index_global' not in json_data:
                (tc_text, tc_path) = json_data
            else:
                # Import a new use_case_step that includes the following:
                use_case_step = useCaseStep(json_data['index_global'], json_data['step'],
                                            json_data['parameter'], json_data['returns'], uc_number, step_number)
                step_number += 1  # The index of the next step.

                use_case.add_ts(use_case_step, use_case_step.step)

                # Extract Action-Object.
                action_object = extract_action_object(use_case_step.step)

                use_case_step.action = action_object[0]
                use_case_step.object.append(action_object[1])

                if len(use_case_step.object[0]) == 0 and len(use_case_step.action) != 0:
                    use_case_step.object[0] = use_case_step.action  # Many ROOTs in the dictionary model are actually obj's, but are placed in the action field.

                if len(use_case_step.action) == 0 or len(use_case_step.object[0]) == 0:
                    print(use_case_step.action, use_case_step.object, use_case_step.index_global)

                # The latter judgement of duplicity basically uses .step_clean, which removes unnecessary modifiers.
                # Too much modifiers can increase the similarity of act and obj.
                use_case_step.step_clean = use_case_step.action + str(use_case_step.object[0])



    return use_case_list

# use Stanford tool to extract.
def extract_from_ucText(use_case_list):
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', lang='zh', memory='8g')

    for uc in use_case_list:
        # 0. First remove special symbols from path and text, then connect with commas
        tc_path = uc.tc_path.replace('/', ',')
        tc_path = re.sub(r'[\d_\\]+', '', tc_path)
        tc_text = uc.tc_text.replace('/', ',')
        tc_text = re.sub(r'[\d_\\]+', '', tc_text)

        # 1. First extract keywords from tc_text
        dict = nlp.pos_tag(tc_text)
        for str, tag in dict:
            # NT is a time noun, VA is a predicative adjective
            if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA':
                uc.keyword.append(str)

        # (Optional) The second part can be commented out if tc_path keywords are not needed. But for now, keep consistent with pub dataset, requiring both uc path / dataset
        # 2. Then extract verbs and nouns from tc_path as supplements
        dict = nlp.pos_tag(tc_path)
        for str, tag in dict:
            if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA':
                uc.keyword.append(str)

        # 3. (If using the recommended measurement method, deduplication is needed; otherwise it's optional) Deduplicate, because when recommending the top k later, if there are duplicates, the actual number of recommendations will be less
        # uc.keyword = [item for i, item in enumerate(uc.keyword) if item not in uc.keyword[:i]]

    nlp.close()
    return use_case_list

def re_extract_act_obj(use_case_list):
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', lang='zh', memory='8g')
    # Some entries in the user dictionary are fixed collocations; some are only act or obj items
    user_dict = {'校验': '状态', '检查': '检查', '预处理': '预处理', '冷复位': '冷复位',
                 '硬复位': '网元', '配置': '路由', '相同': '相同', '日志': '日志', '清理': '环境',
                 '准备': '测试', '构造': '数据', '导出': '数据', '初始化': '参数', '对比': '对比', '一致': '一致',
                 '单板': '单板', '复位': '复位', '准备': '准备', '取消': '预留', '关闭': '连接', '时间戳': '时间戳',
                 '资源': '资源', '返回': '返回', '比较': '比较'}

    for uc in use_case_list:
        # print('Current ucid:', uc.index_global)
        for step in uc.step_list:
            step.action = ''
            step.object = []  # Reset the original BFGen extraction results
            va = ''  # Reset the previous VA (adverb)
            # First look for dobj relation from dependency relations, if there is a direct object dobj, use it directly
            tokens = nlp.word_tokenize(step.step)
            dependency = nlp.dependency_parse(step.step)
            for item in dependency:
                if item[0] == 'dobj':
                    step.action = tokens[item[1] - 1]
                    step.object = [tokens[item[2] - 1]]
                    break

            # If there is no dobj, look for verbs
            if not step.action:
                tags = nlp.pos_tag(step.step)
                for str, tag in tags:
                    if not step.action and tag.startswith(('V')) and tag != 'VA':  # VA is a predicative adjective
                        step.action = str
                    elif not step.object and tag.startswith(('N')) and tag != 'NT':  # NT is a time noun
                        step.object.append(str)
                    elif tag == 'VA':
                        va = str  # Record va, use va when there is no dobj

            if not step.action or not step.object:
                # In special cases where there is still a missing item, look in the user dictionary
                if '校验' in step.step:
                    step.action = '校验'
                    if '状态' in step.step:
                        step.object = ['状态']  # '校验' pairs with '状态'
                if not step.action:
                    for key, value in user_dict.items():
                        if key in step.step:
                            step.action = key
                            if step.object and step.object[0] == step.action:
                                step.object[0] = value
                elif step.action and not step.object and va:
                    step.object = [va]

            if not step.action:  # If there are still cases where act does not exist, take a word other than obj
                for str, tag in tags:
                    if step.object and str != step.object[0]:
                        step.action = str
                        break
            if not step.object:
                step.object.append(step.action)  # Many statements just don't have obj

            # Check if there are still no act/obj
            if not step.action or not step.object:
                print("test step is not have action or obj:", step.index_global, tags)

    nlp.close()
    return use_case_list

def eval_general_baseline_ncet(use_case_list):
    out_dict = {"p_record": [], "r_record": [], "f1_record": [], "auc_record": [],
                "p_ave": 0, "r_ave": 0, "f1_ave": 0, "auc_ave": 0, "dataset": ""}

    # 1. Calculate metrics for each uc
    for uc in use_case_list:
        ground_truth, tp = [], []
        out_dict['dataset'] = uc['dataset']

        # 1. Collect ground truth in each uc, i.e., act/obj in steps
        for step in uc.step_list:
            ground_truth.append(step.action)
            ground_truth.append(step.object[0])
        ground_truth = [word.lower() for word in ground_truth]  # Convert each English letter to lowercase

        # 2. Find tp (uc.keyword is pred node by rule_based method)
        ground_truth_copy = ground_truth[:]  # Make a copy for searching tp, delete existing ones to prevent duplicate counting
        for pred_node in uc.keyword:
            pred_node = pred_node.lower()  # Convert each English letter to lowercase
            if pred_node in ground_truth_copy:
                tp.append(pred_node)
                ground_truth_copy.remove(pred_node)

        # 3.1. auc
        auc = (0.5 * len(tp)) / len(ground_truth)  # Formula from metric_auc_llm_match
        out_dict["auc_record"].append(auc)

        # 3.2. precision
        p_value = len(tp) / (len(uc.keyword)) if (len(uc.keyword)) > 0 else 0  # From metric_precision_strict_llm_match
        out_dict["p_record"].append(p_value)

        # 3.3. recall
        recall_value = len(tp) / (len(ground_truth)) if len(ground_truth) > 0 else 0  # From metric_recall_llm_match
        out_dict["r_record"].append(recall_value)

        # 3.4. f1
        if (p_value + recall_value) != 0:
            f1_value = (2 * p_value * recall_value) / (p_value + recall_value)
        else:
            f1_value = 0
        out_dict["f1_record"].append(f1_value)

    # 2. Calculate average metrics for the dataset
    out_dict["p_ave"] = sum(out_dict["p_record"]) / len(out_dict["p_record"])
    out_dict["r_ave"] = sum(out_dict["r_record"]) / len(out_dict["r_record"])
    out_dict["f1_ave"] = sum(out_dict["f1_record"]) / len(out_dict["f1_record"])
    out_dict["auc_ave"] = sum(out_dict["auc_record"]) / len(out_dict["auc_record"])

    return out_dict


if __name__ == '__main__':
    task_index = 1
    print(f"******** start # {task_index} task !! start time：{datetime.now().strftime('%H:%M:%S')}********")


    # task1:(NCE-T)metric with p/r/f1/auc
    if task_index == 1:
        # Save path
        out_path = "ControlledExper/5_experiment_data/1_result/1_baseline/3_general_method/NCET/nce-t.json"

        # 1. Extract the use_case_list from the .json file, mainly using the tc_text and tc_path.
        use_case_list = extract_use_case_step_1("data/NCE-T_DATASET/NCE-T_tc.json")
        # 2. Verbs and objects are extracted in tc_text.
        #  all words are extracted in tc_path. Stored in uc.keyword. keyword is "core word" in paper.
        use_case_list = extract_from_ucText(use_case_list)
        # 3. Extract the act/obj in the uc step with the Stanford method.
        use_case_list = re_extract_act_obj(use_case_list)
        # 4. Evaluate with p/r/f1/auc
        out_dict = eval_general_baseline_ncet(use_case_list)
        # 5、Save the evaluation result to a json file.
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(f"ave Precision: {out_dict['p_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave Recall: {out_dict['r_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave F1 Score: {out_dict['f1_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave AUC: {out_dict['auc_ave']:.3f}", ensure_ascii=False) + '\n\n')
            for index in range(len(out_dict["p_record"])):
                f.write(json.dumps(f"index: {index},  P: {out_dict['p_record'][index]:.3f}, ", ensure_ascii=False))
                f.write(json.dumps(f" R: {out_dict['r_record'][index]:.3f}, ", ensure_ascii=False))
                f.write(json.dumps(f" F1: {out_dict['f1_record'][index]:.3f}, ", ensure_ascii=False))
                f.write(json.dumps(f" AUC: {out_dict['auc_record'][index]:.3f}", ensure_ascii=False) + '\n')


    print("******** finish ********")