# Read UC and store in dict
def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # line = line.replace("'", '"')  # Sometimes there are extra single or double quotes in the dataset
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"Error message: {e},  Line: {inspect.currentframe().f_lineno}, json reading failed")
    return use_case_list

# Specifically for English dataset pub dataset
def extract_from_ucText_eng(use_case_list):
    # No lang=zh annotation here, default is English
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', memory='8g')

    for uc in use_case_list:  # The original keys in the data are all extracted by llm, here we need to re-extract them using stf method
        uc['key_name'], uc['key_path'], uc['key_act'], uc['key_obj'] = [], [], [], []

        # 0. Split camel case naming in uc name, extract verbs and nouns
        uc_name = re.sub(r'(?=[A-Z])', ' ', uc["ucName"])
        uc_name = uc_name.split()
        for sent in uc_name:
            dict = nlp.pos_tag(sent)
            for str1, tag in dict:
                # Collect all nouns and verbs
                if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA':  # NT is time noun, VA is predicative adjective
                    uc['key_name'].append(format_node_eng(str1))  # Convert to lowercase
        if len(uc['key_name']) == 0:  # If there are no verbs or nouns in the name, add the entire name
            uc['key_name'].append(format_node_eng(uc["ucName"]))

        # 1. Add dataset as a key, equivalent to uc path
        uc_dataset = uc['dataset']
        uc['key_path'].append(format_node_eng(uc_dataset))

        # 2. Split uctext by periods and line breaks
        if "uctext" not in uc or uc['uctext'] == 'None':
            continue
        uctext = None
        if "uctext" in uc and isinstance(uc['uctext'], str):
            if "i.e." in uc['uctext']:
                uc['uctext'] = uc['uctext'].replace("i.e.", "ie ")
            if "e.g." in uc['uctext']:
                uc['uctext'] = uc['uctext'].replace("e.g.", "eg ")
            uctext = re.split(r'[.\n]', uc['uctext'])
            uctext = [s for s in uctext if s.strip()]  # Remove empty lines

        # 1. Extract keywords from uctext, store in uc['keyword']
        if not uctext and "uctext" in uc and uc['uctext'] != 'None':
            uctext = uc['uctext']
        for sent in uctext:
            dict = nlp.pos_tag(sent)
            for str2, tag in dict:
                # Collect all nouns and verbs
                if tag.startswith(('V')) and tag != 'NT' and tag != 'VA':  # NT is time noun, VA is predicative adjective
                    if format_node_eng(str2):
                        uc['key_act'].append(format_node_eng(str2))
                elif tag.startswith(('N')) and tag != 'NT' and tag != 'VA':  # NT is time noun, VA is predicative adjective
                    if format_node_eng(str2):
                        uc['key_obj'].append(format_node_eng(str2))

        # 3. (If using the recommended measurement method, need to deduplicate; otherwise not necessary) Deduplicate, because when recommending the top k, if there are duplicates, it will result in fewer actual recommendations
        # uc['keyword'] = [item for i, item in enumerate(uc['keyword']) if item not in uc['keyword'][:i]]

    nlp.close()
    return use_case_list

def re_extract_act_obj_eng(use_case_list):
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', memory='8g')
    for uc in use_case_list:
        uc['act'], uc['obj'],act_list,obj_list = [], [],[], []   # Reset the original BFGen extraction results
        for step_list in uc['steps']:
             for step in step_list:
                if contains_only_digits_symbols_spaces(step):  # If the sentence contains no letters, only numbers, symbols, and spaces
                    continue
                va, action, object = None, [], []  # Reset the previous one (VA (adverb))
                # First look for dobj relations from dependency relations, if there is a direct object dobj, use it directly
                tokens = nlp.word_tokenize(step)
                dependency = nlp.dependency_parse(step)
                for item in dependency:
                    if item[0] == 'dobj':
                        action.append(tokens[item[1] - 1])
                        object.append([tokens[item[2] - 1]])
                        break

                # If there is no dobj, then look for verbs
                if len(action) == 0:
                    tags = nlp.pos_tag(step)
                    for str, tag in tags:
                        if tag.startswith(('V')) and tag != 'VA':  # VA is predicative adjective
                            action.append(str)  # pub dataset is informal, there may be multiple actions in one sentence, so collect all.
                        elif tag.startswith(('N')) and tag != 'NT':  # NT is time noun
                            object.append(str)
                        elif tag == 'VA':
                            va = str  # Record va, use va when there is no dobj

                if len(action) != 0 and len(object) == 0 and va:
                    object.append(va)

                if len(action) == 0:  # If there are still cases where act does not exist, take a word other than obj
                    for str, tag in tags:
                        if object and str not in object:
                            action.append(str)
                            break
                if len(object) == 0:
                    if len(action) == 0:
                        action.append(max(step.split(), key=len))
                    object.append(action[0])  # Many statements just don't have obj, e.g.: manual return
                if len(action) == 0:  # In many sentences, act is mistaken for obj, e.g.: 'User types Master Password'. Find an item in the sentence that is not capitalized.
                    action = [item for item in object if item.islower()]

                # If there are still missing act/obj
                if len(action) == 0 or len(object) == 0:
                    if len(action) == 0:
                        action = object
                    elif len(object) == 0:
                        object = action
                    else:
                        print("test step is not have action or obj:", step)

                if len(action) == 0 or len(object) == 0:  # Final check
                    print("test step is not have action or obj:", step)

                aaa = formate_node_eng_list(action)
                if aaa:
                    act_list.append(aaa[-1])

                ooo = formate_node_eng_list(object)
                if ooo:
                    obj_list.append(ooo[-1])

             uc['act'].append(act_list)
             uc['obj'].append(obj_list)
             act_list,obj_list = [],[]

    nlp.close()
    return use_case_list

def eval_general_baseline_pub(use_case_list):
    out_dict = {"p_record": [], "r_record": [], "f1_record": [], "auc_record": [],
                "p_ave": 0, "r_ave": 0, "f1_ave": 0, "auc_ave": 0, "dataset": ""}

    # 1. Calculate metrics for each uc
    for uc in use_case_list:
        ground_truth, tp = [], []
        out_dict['dataset'] = "pub dataset"

        # 1. Collect ground truth in each uc, i.e., act/obj in steps
        ground_truth += uc['act']
        ground_truth += uc['obj']
        ground_truth = [word.lower() for word in ground_truth]  # Convert each English letter to lowercase

        # 2. Find tp (uc.keyword is pred node)
        ground_truth_copy = ground_truth[:]  # Make a copy to facilitate searching for tp, delete existing ones to prevent duplicate counting
        for pred_node in uc['keyword']:
            pred_node = pred_node.lower()  # Convert each English letter to lowercase
            if pred_node in ground_truth_copy:
                tp.append(pred_node)
                ground_truth_copy.remove(pred_node)

        # 3.1. auc
        auc = (0.5 * len(tp)) / len(ground_truth)  # Formula from metric_auc_llm_match
        out_dict["auc_record"].append(auc)

        # 3.2. precision
        p_value = len(tp) / (len(uc["keyword"])) if (len(
            uc["keyword"])) > 0 else 0  # From metric_precision_strict_llm_match
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
    task_index = 2

    # task2:pub,rule_based method with p/r/f1/auc
    if task_index == 2:
        in_path = "ControlledExper/4_dataset_pred_node/ERNIE_4_Turbo_8k/with_tp/2nd_round/step_seg/Ernie_pub_seg.json"
        out_path = "ControlledExper/5_experiment_data/1_result/1_baseline/3_general_method/pub_dataset/pub.json"

        # 1. Read uc list
        use_case_list = read_uc_from_json(in_path)

        # 2. Use Stanford to extract key nodes (uc name\uc text). Note: The key nodes in the previous file were extracted using llm.
        use_case_list = extract_from_ucText_eng(use_case_list)
        # 3. Use the current method to extract act/obj from uc steps. (The ones generated in the first step were extracted using the spacy method, now re-extract using the Stanford method)
        use_case_list = re_extract_act_obj_eng(use_case_list)
        # 4. Based on uc.keyword and subsequent uc content, determine the accuracy of the traditional method, measured using p/r/f1/auc
        out_dict = eval_general_baseline_pub(use_case_list)

        # 5. Save locally
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(f"ave Precision: {out_dict['p_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave Recall: {out_dict['r_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave F1 Score: {out_dict['f1_ave']:.3f}", ensure_ascii=False) + '\n')
            f.write(json.dumps(f"ave AUC: {out_dict['auc_ave']:.3f}", ensure_ascii=False) + '\n\n')
            for index in range(len(out_dict["p_record"])):
                f.write(
                    json.dumps(f"index: {index},  P: {out_dict['p_record'][index]:.3f}, dataset: {out_dict['dataset']}",
                               ensure_ascii=False))
                f.write(json.dumps(f" R: {out_dict['r_record'][index]:.3f}, dataset: {out_dict['dataset']}",
                                   ensure_ascii=False))
                f.write(json.dumps(f" F1: {out_dict['f1_record'][index]:.3f}, dataset: {out_dict['dataset']}",
                                   ensure_ascii=False))
                f.write(json.dumps(f" AUC: {out_dict['auc_record'][index]:.3f}, dataset: {out_dict['dataset']}",
                                   ensure_ascii=False) + '\n')

    print("******** finish ********")