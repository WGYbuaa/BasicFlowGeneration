# 从另外的json文档中，提取 items ，合并到其他use_case_list 中
def get_keys_from_json(use_case_list, baseline_uc_list, key):
    if len(use_case_list) != len(baseline_uc_list):
        print(f'两个 use_case_list 中 uc 个数不统一！！')
    for uc, uc_base in zip(use_case_list, baseline_uc_list):
        if key not in uc.keys() and key in uc_base.keys():
            uc[key] = uc_base[key]
    return use_case_list

# 读取用dict保存的uc
def read_uc_from_json(file_path):
    use_case_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # line = line.replace("'", '"')  # 有时候数据集中会有多余的单引号或者双引号
            try:
                uc = json.loads(line)
                use_case_list.append(uc)
            except json.JSONDecodeError as e:
                print(f"错误信息: {e},  Line: {inspect.currentframe().f_lineno}, json读取失败")
    return use_case_list

def format_node_eng(input1):
    update1 = []
    if isinstance(input1, list):
        for item1 in input1:
            if item1 != "" and not item1.isdigit() and not is_pure_punctuation(item1):
                item1 = item1.strip()  # 去除两端空格
                item1 = re.sub(r'^[\W_]+|[\W_]+$', '', item1)  # 去除两端特殊符号
                item1 = item1 #.lower()  # 统一大小写
                update1.append(item1)
    elif isinstance(input1, str):
        if input1 != "" and not input1.isdigit() and not is_pure_punctuation(input1):
            item2 = input1.strip()  # 去除两端空格
            item2 = re.sub(r'^[\W_]+|[\W_]+$', '', item2)  # 去除两端特殊符号
            update1 = item2 #.lower()  # 统一大小写
    else:
        print(f'类型 出错, {input1}')

    if update1 == []:
        return None
    return update1

# 英文数据集pub dataset专用
def extract_from_ucText_eng(use_case_list):
    # 这里不标注lang=zh，默认英文
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', memory='8g')

    for uc in use_case_list:  # 数据中原本的key都是llm提取的，这里该用stf方法重新提取一次
        uc['key_name'], uc['key_path'], uc['key_act'], uc['key_obj'] = [], [], [], []

        # 0、将uc name中驼峰命名法分开,提取其中的动词和名词。
        uc_name = re.sub(r'(?=[A-Z])', ' ', uc["ucName"])
        uc_name = uc_name.split()
        for sent in uc_name:
            dict = nlp.pos_tag(sent)
            for str1, tag in dict:
                # 收集其中所有的名词和动词
                if tag.startswith(('V', 'N')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    uc['key_name'].append(format_node_eng(str1))  # 都变为小写
        if len(uc['key_name']) == 0:  # 如果name中没有动、名词，则将整个name加入。
            uc['key_name'].append(format_node_eng(uc["ucName"]))

        # 1、将 dataset 也作为key加入其中，相当于 uc path。
        uc_dataset = uc['dataset']
        uc['key_path'].append(format_node_eng(uc_dataset))

        # 2、将uctext依据句点和换行符进行断句。
        if "uctext" not in uc or uc['uctext'] == 'None':
            continue
        uctext = None
        if "uctext" in uc and isinstance(uc['uctext'], str):
            if "i.e." in uc['uctext']:
                uc['uctext'] = uc['uctext'].replace("i.e.", "ie ")
            if "e.g." in uc['uctext']:
                uc['uctext'] = uc['uctext'].replace("e.g.", "eg ")
            uctext = re.split(r'[.\n]', uc['uctext'])
            uctext = [s for s in uctext if s.strip()]  # 去除空格行

        # 1、提取 uctext 中的关键词,存放在 uc['keyword']
        if not uctext and "uctext" in uc and uc['uctext'] != 'None':
            uctext = uc['uctext']
        for sent in uctext:
            dict = nlp.pos_tag(sent)
            for str2, tag in dict:
                # 收集其中所有的名词和动词
                if tag.startswith(('V')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    if format_node_eng(str2):
                        uc['key_act'].append(format_node_eng(str2))
                elif tag.startswith(('N')) and tag != 'NT' and tag != 'VA':  # NT是时间名词, VA是表语形容词
                    if format_node_eng(str2):
                        uc['key_obj'].append(format_node_eng(str2))

        # 3、(如果用推荐的度量方法，需要去重；其余可以不用)去重，因为后面推荐的时候前k个，如果有重复，则会导致实际推荐个数少
        # uc['keyword'] = [item for i, item in enumerate(uc['keyword']) if item not in uc['keyword'][:i]]

    nlp.close()
    return use_case_list

def re_extract_act_obj_eng(use_case_list):
    nlp = StanfordCoreNLP(r'ControlledExper/stanford-corenlp-4.0.0', memory='8g')
    for uc in use_case_list:
        uc['act'], uc['obj'],act_list,obj_list = [], [],[], []   # 清零原本ASSAM的提取结果
        for step_list in uc['steps']:
             for step in step_list:
                if contains_only_digits_symbols_spaces(step):  # 如果句子中没有字母，只有数字、符号和空格等
                    continue
                va, action, object = None, [], []  # 清零上一个(VA（副词）)
                # 先从依存关系寻找dobj关系，如果有直接宾语dobj，则直接用
                tokens = nlp.word_tokenize(step)
                dependency = nlp.dependency_parse(step)
                for item in dependency:
                    if item[0] == 'dobj':
                        action.append(tokens[item[1] - 1])
                        object.append([tokens[item[2] - 1]])
                        break

                # 如果没有dobj，则从动词中寻找
                if len(action) == 0:
                    tags = nlp.pos_tag(step)
                    for str, tag in tags:
                        if tag.startswith(('V')) and tag != 'VA':  # VA是表语形容词
                            action.append(str)  # NCET因为是rtcm，一句话只有一个动作。pub数据集不是严格rtcm，一句话中会有多个动作，所以全部收集。
                        elif tag.startswith(('N')) and tag != 'NT':  # NT是时间名词
                            object.append(str)
                        elif tag == 'VA':
                            va = str  # 记录va，在没有dobj的情况下使用va

                if len(action) != 0 and len(object) == 0 and va:
                    object.append(va)

                if len(action) == 0:  # 如果还是有不存在act的情况，则取一个除obj外的词
                    for str, tag in tags:
                        if object and str not in object:
                            action.append(str)
                            break
                if len(object) == 0:
                    if len(action) == 0:
                        action.append(max(step.split(), key=len))
                    object.append(action[0])  # 很多语句就是没有obj，例如：人工返回
                if len(action) == 0:  # 很多句子中act被误认为是obj，例如：'User types Master Password'。找一个句子中一个没有大写的项。
                    action = [item for item in object if item.islower()]

                # 如果还是有确实act/obj的
                if len(action) == 0 or len(object) == 0:
                    if len(action) == 0:
                        action = object
                    elif len(object) == 0:
                        object = action
                    else:
                        print("test step is not have action or obj:", step)

                if len(action) == 0 or len(object) == 0:  # 最终
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

def check_argc(use_case_list_dict):
    # 1.1 统计所有的node数目（act\obj\key）
    if isinstance(use_case_list_dict[0]["act"][0], list):  # 如果是嵌套列表，即node已经按照step分好了。（chatgpt版本此处已经分好）
        argc = count_node_data_dict_all_nested(use_case_list_dict)
    elif isinstance(use_case_list_dict[0]["act"][0], int):  # 如果不是，则node没有按照step分好。（Ernie版本此处还不需要）
        argc = count_node_data_dict_all(use_case_list_dict)
    else:
        print(f'ERROR: check_argc!!')
    # # 1.2 为了统计edge数目，生成一次全局edge_list
    # edge_dict, node_to_UCText = create_edges_llm(use_case_list_dict, ERROR_WORD_LIST, use_case_list_ori)
    #
    # act, obj, key, edges_num, edges_deduplicate = count_node_sub_data(edge_dict)
    # if len(act) != argc['act_node']:
    #     print("act node 全局数目对不上！！")
    # if len(obj) != argc['obj_node']:
    #     print("obj node 全局数目对不上！！")
    # if len(key) != argc['key_node']:
    #     print("key node 全局数目对不上！！")
    # argc['edges_num'] = edges_num
    # argc['edges_deduplicate'] = edges_deduplicate
    return argc


def group_uc_fixed(use_case_list, group_uc):
    use_case_list_new = []
    for index_list in group_uc:
        sub_uc_list = []
        for index in index_list:
            sub_uc_list.append(use_case_list[index])
        use_case_list_new.append(sub_uc_list)
    return use_case_list_new

def get_edges_dict_and_node_to_UCText_only_rgat(use_case_list):
    edges_dict_list = []
    node_to_UCText_list = []
    for uc_list in use_case_list:
        edge_dict, node_to_UCText = create_edges_llm_only_rgat(uc_list)
        edges_dict_list.append(edge_dict)
        node_to_UCText_list.append(node_to_UCText)

    return edges_dict_list, node_to_UCText_list

# 统计节点数量（入参为dict）, 根据子图统计
def count_node_data_dict_sub(edges_dict_list):
    node_total, node_max_sub_data = 0, 0
    act_node_num, obj_node_num, key_node_num = [], [], []  # 不同子图会包含相同的点，所以需要去重
    edge_num_total = 0
    edges_deduplicate_all = []
    for edges_list in edges_dict_list:
        print(f'edge_list index: {edges_dict_list.index(edges_list)}')
        # 1. 计算每个子图中各种点的个数 (这里去重的edge数没意义（edges_deduplicate），因为只是每个子图的去重，整体还是会有重复)
        act, obj, key, edges_num, edges_deduplicate = count_node_sub_data(edges_list)

        # 2. 记录所有子图中最多node(act+obj+key)数目
        if (len(act) + len(obj) + len(key)) > node_max_sub_data:
            node_max_sub_data = (len(act) + len(obj) + len(key))

        # 3. 统计各个类别的node总数
        act_node_num.extend(set(act) - set(act_node_num))
        obj_node_num.extend(set(obj) - set(obj_node_num))
        key_node_num.extend(set(key) - set(key_node_num))

        # # 4. 累加 edge 总数
        # edge_num_total += edges_num
        #
        # # 5. edge 去重的总数
        # edges_deduplicate_all = find_diff_list(edges_deduplicate_all, edges_deduplicate)

    # 6. 累加所有node(act+obj+key)数目
    node_total = len(act_node_num) + len(obj_node_num) + len(key_node_num)

    print(
        f'node总数: {node_total}, act总数: {len(act_node_num)}, obj总数: {len(obj_node_num)},'
        f' key总数: {len(key_node_num)}, 子图中最多node数: {node_max_sub_data}')
    # f' edge总数: {edge_num_total}, edge 去重后总数：{len(edges_deduplicate_all)}.')

    return {'node_total': node_total, 'act_node': len(act_node_num), 'obj_node': len(obj_node_num),
            'key_node': len(key_node_num),
            'node_max_sub': node_max_sub_data, 'edge_num_total': edge_num_total}
    # 'edge_dedup': len(edges_deduplicate_all)

# 生成数据集,修改自_6_multi_Data_makeDataset 的 generate_dataset_1()
# 这个为多数据集版本，且包含x, edge_index, edge_type, edge_attr、y、train_mask
def generate_dataset_4turbo(edges_dict, node_to_UCText, para_dict):
    # 1、获得边的重复度，之后可用作权重
    for key in edges_dict.keys():
        edges_dict[key] = [(item, count) for item, count in Counter(edges_dict[key]).items()]

    # 2、节点到编号的映射(编号是当前子图中节点的编号，从0开始)
    node_to_id = {}
    id_to_node = {}

    # 遍历边数据以建立节点到编号的映射和收集节点类型
    next_id = 0
    for source_node, edges in edges_dict.items():
        source_node_str = f"{source_node[0]}_{source_node[1]}"  # 节点名称+类型作为唯一标识，因为可能用名称相同，但是类型不同的节点。如果仅用名称为标识，则可能出错
        if source_node_str not in node_to_id:
            node_to_id[source_node_str] = next_id
            id_to_node[next_id] = source_node_str
            next_id += 1
        for target_node, times in edges:
            target_node_str = f"{target_node[0]}_{target_node[1]}"
            if target_node_str not in node_to_id:
                node_to_id[target_node_str] = next_id
                id_to_node[next_id] = target_node_str
                next_id += 1

    # 3、创建节点特征embedding
    node_embeddings = []
    for node_id in id_to_node:
        node_str = id_to_node[node_id]
        # 分割节点名称和类型
        node_name, node_type = node_str.rsplit('_', 1)
        # 获取嵌入
        embedding = MODEL.encode([node_name])[0]
        embedding = torch.from_numpy(embedding)  # 将这个 NumPy 数组转换为 PyTorch 张量
        node_embeddings.append(embedding)

    # 将嵌入转换为PyTorch张量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.stack(node_embeddings, dim=0).to(device, dtype=torch.float)

    # 4、创建边数据
    edge_index = []
    edge_attr = []  # 边权重
    edge_type = []  # 用于存储边的类型

    for source_node, edges in edges_dict.items():
        source_id = node_to_id[f"{source_node[0]}_{source_node[1]}"]
        for target_node, weight in edges:
            target_id = node_to_id[f"{target_node[0]}_{target_node[1]}"]
            edge_index.append([source_id, target_id])
            edge_attr.append(weight)

            # 确定边的类型
            if source_node[1] == 'act' and target_node[1] == 'act':
                edge_type.append(0)  # 假设0表示act-act边
            elif source_node[1] == 'act' and target_node[1] == 'obj':
                edge_type.append(1)  # 假设1表示act-obj边
            elif source_node[1] == 'obj' and target_node[1] == 'obj':
                edge_type.append(2)  # 假设2表示obj-obj边
            elif source_node[1] == 'keyword' and target_node[1] == 'act':
                edge_type.append(3)  # 假设3表示 keyword-act 边
            elif source_node[1] == 'keyword' and target_node[1] == 'obj':
                edge_type.append(4)  # 假设4表示 keyword-obj 边
            elif source_node[1] == 'keyword' and target_node[1] == 'keyword':
                edge_type.append(5)  # 假设5表示 keyword-keyword 边
            else:
                print(source_node[1], target_node[1])
                print(f'edge_type Error!!!')
                print(source_node[1], target_node[1])

    # 将edge_index和edge_type转换为PyTorch张量
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_index_reverse = edge_index[[1, 0], :]  # 将edge的index反转，等于是将原本有向边变为无向边，因为两个方向的edge都有了
    edge_index = torch.cat((edge_index, edge_index_reverse), dim=1)  # 然后直接拼接在后面

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)  # 权重是出现次数，所以使用int
    edge_attr = edge_attr.view(-1, 1)  # 转换为二维，形状为 [num_edges, 1]
    edge_attr = torch.cat((edge_attr, edge_attr), dim=0)  # 有向边变成无向边，权重直接复制一份放在后面即可

    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_type = torch.cat((edge_type, edge_type), dim=0)  # 有向边变成无向边，类型也是直接复制一份放在后面即可

    # 5、生成非预定义参数（不能用y的原因是Data对象要求y必须是[节点数]的向量，形状不能变化）
    # 5.1、生成标签 y，是一个列表，顺序对应keyword，存放每个keyword关联的act/obj的id
    y = []  # 顺序按照node_to_id中keyword的顺序
    keyword_id = []  # 记录keyword节点的id
    for key_node in node_to_id:
        if 'keyword' in key_node:
            keyword_id.append(node_to_id[key_node])
            related_node = []
            node_name, node_type = key_node.rsplit('_', 1)
            for (tar_node_str, target_node_type), time in edges_dict[(node_name, node_type)]:
                if target_node_type != 'keyword':
                    tar_node = f"{tar_node_str}_{target_node_type}"
                    related_node.append(node_to_id[tar_node])
            y.append(related_node)
        else:
            y.append([-1])  # 除了keyword之外的节点位置补充-1

    # 确定最大的列表长度为 节点总数,需要所有列表长度一致
    max_length = para_dict['max_node_subdata']  # 所有子图最多节点数原为1257
    # 创建一个填充后的二维列表，填充'-99'表示无效数据，补齐列表
    padded_y = [sublist + [-99] * (max_length - len(sublist)) for sublist in y]
    # 将填充后的列表转换为张量
    y_tensor = torch.tensor(padded_y, dtype=torch.long)

    # # 5.2、生成训练掩码train_mask, 用整张子图（Data对象）来训练，因为抽百分比的数据的话，数据集会不平衡（各种类型的点数量不一致）
    # # 计算训练节点的数量
    # train_mask = torch.zeros(len(id_to_node), dtype=torch.bool)
    # train_mask[keyword_id] = True

    # 6、转变node_to_UCText为非预定义参数
    node_to_UCText_new = {}  # 创建一个新的用于存放
    for node in node_to_UCText.keys():
        node_id = node_to_id[node]
        node_to_UCText_new[node_id] = node_to_UCText[node]  # node_to_UCText_new中item是 节点id：uctext id
        node_to_UCText_new[node_id] = list(set(node_to_UCText_new[node_id]))  # 列表去重

    sort_key = sorted(node_to_UCText_new.keys())  # 按照节点id排序
    padded_UC = [[item + para_dict['uctext_start'] for item in node_to_UCText_new[key]] for key in
                 sort_key]  # 按照节点id排序 并从总节点id (原来为)3241后开始计算UCText的id

    # 填充无效数据'-99'至最长维度，这里的最长长度max_length变成了众多子图中最多的UC数目
    max_length = para_dict['max_length']  # 原为1744
    # 遍历result中的每个小列表
    for i in range(len(padded_UC)):
        # 计算当前小列表需要的填充长度
        padding_length = max_length - len(padded_UC[i])
        # 如果需要填充，则用-99填充到长度为max_length
        if padding_length > 0:
            padded_UC[i] += [-99] * padding_length

    for sublist in padded_UC:  # 检查长度是否补齐
        if len(sublist) != max_length:
            print(f"子列表 {sublist} 的长度不等于最长{max_length}，其id为：{padded_UC.index(sublist)}")
    node_to_UCText_new = torch.tensor(padded_UC, dtype=torch.long)

    # 7、将数据添加到Data对象中,包含x、edge_index、edge_attr和edge_attr
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_type=edge_type)
    data.keyword_relation = y_tensor
    data.to_UCText = node_to_UCText_new  # 表示 每个node存在于哪个UCText里，节点id：UCText_id

    return data

if __name__ == "__main__":
    NECT_groud_truth_path = "/20240421/ControlledExper/2_dataset_origin_node/Ernie_NEC-T_groud_truth.json"
    NECT_groud_truth_reextract = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_reextract.json"
    NECT_groud_truth_with_keyword = "E:/bertTest/20240421/ControlledExperiment_conventionAlgorithm/2_dataset_origin_node/Ernie_NEC-T_groud_truth_with_keyword.json"

    print(f' 程序开始时间：{datetime.now().strftime("%H:%M:%S")}')
    # 确认要完成的 task_index
    task_index = 2  # 选择需要完成的task
    # task5:(pub,pt,only rgat)制作pt文件。仿照上述task4
    if task_index == 5:
        in_file = "E:/GitHub/ASSAM/data/2_dataset_origin_node/Ernie-4-Turbo/2_pub_after_formalized/Ernie_pub_gt.json"
        pt_save_path = "ControlledExper/5_experiment_data/2_pt_file/only_rgat/2nd_after_formalized/pub_only_rgat_20.pt"
        ucname_file = "E:/GitHub/ASSAM/data/4_dataset_pred_node/chatgpt_4o/2nd_round/with_tp/ChatGPT_pub.json"

        # 1、读取文件
        use_case_list = read_uc_from_json(in_file)
        uc_base = read_uc_from_json(ucname_file)
        use_case_list = get_keys_from_json(use_case_list, uc_base, 'ucName')

        index = 0
        # 2、删除其他方法提取的node
        for uc in use_case_list:
            uc['key_path'] = format_node_eng(uc['dataset'])
            for key in ["pred_steps", "pred_act", "pred_obj", "tp_act", "tp_obj"]:  # 删除一些 不然内存不够
                if key in uc.keys():
                    del uc[key]
            if "key_act" not in uc.keys():
                uc['key_act'] = []
            if "key_obj" not in uc.keys():
                uc['key_obj'] = []
            uc['index'] = index
            index += 1

        # 3、用Stanford方法提取key node、act/obj node，用于后续制作pt文件（仿照 _controlled_exper_NLP&Rule-based.py task2）
        # 3.1、用Stanford提取key node（uc name\uc text)。
        use_case_list = extract_from_ucText_eng(use_case_list)
        # 3.2、用当前方法提取uc step中的act/obj。
        use_case_list = re_extract_act_obj_eng(use_case_list)
        # 3.3 生成全局的边数目和node数目
        argc1 = check_argc(use_case_list)

        # # 4、进行分组
        # use_case_list = group_pub_uc(use_case_list, ARGC_20['max_uc_in_sub'])
        # # 4.1 打印分组情况，用于后续固定分组
        # grouping_index = []
        # for uc_list in use_case_list:
        #     grouping_index_sub = []
        #     for uc in uc_list:
        #         grouping_index_sub.append(uc["index"])
        #     grouping_index.append(grouping_index_sub)
        # print(f"所有uc的分组情况为：{grouping_index}")

        # 4. 因为随机分组时间太久，所以选择保存一份.group_uc_fixed()为固定分组(上面“4、进行分组"至print为随机分组的代码)
        use_case_list = group_uc_fixed(use_case_list, PUB_GROUPING_UC_20_1)

        # 5. 获得两个字典：edges_dict(存放act/obj/keyword之间的edge)；node_to_UCText_list(存放act/obj/keyword to uctext的映射)
        edges_dict_list, node_to_UCText_list = get_edges_dict_and_node_to_UCText_only_rgat(use_case_list)

        # 5.1 统计：所有子图中最多的节点数（act+obj+key）
        argc2 = count_node_data_dict_sub(edges_dict_list)
        # 5.2 判断两次统计是否一致. 除了'node_max_sub'和’edge_num_total‘是后面加的，其他都应该一样
        print(f'全局统计和分图统计不一致的项为：{find_diff_dict(argc1, argc2)}')

        # 6、生成数据集
        dataset = []
        # uctext_start 为uctext节点的起始点（全部数据中act+obj+key节点总数）；max_node_subdata 所有子图最多节点数（act+obj+key）; max_length 所有子图中包含最多uc的个数
        para_dict = {"uctext_start": argc2['node_total'], "max_node_subdata": argc2['node_max_sub'],
                     "max_length": ARGC_20['max_uc_in_sub']}
        for i in range(len(edges_dict_list)):
            subgraph_data = generate_dataset_4turbo(edges_dict_list[i], node_to_UCText_list[i], para_dict)
            dataset.append(subgraph_data)
        # 7、将data保存到文件
        torch.save(dataset, pt_save_path)