from torch_geometric.nn import RGCNConv, RGATConv
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.retrieval import RetrievalMAP, RetrievalAUROC, RetrievalMRR, RetrievalRecall
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC


class RGAT(torch.nn.Module):
    def __init__(self, num_edge_types, in_channels, out_channels, edge_dim, gamma):
        super(RGAT, self).__init__()
        self.gamma = gamma
        self.conv1 = RGATConv(in_channels, out_channels, num_edge_types, heads=4, dropout=0.2,
                              edge_dim=edge_dim, gamma = self.gamma)  # heads数量和 Line15 的out channels 对应
        self.conv2 = RGATConv(out_channels * 4, out_channels, num_edge_types, heads=1, concat=True,
                              dropout=0.2, edge_dim=edge_dim, gamma = self.gamma)  # dropout=0.2/0.3
        self.out = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_type, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        # x = F.normalize(x, p=2, dim=-1)
        x = self.conv1(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # x = F.normalize(x, p=2, dim=-1)
        x = self.conv2(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
        x = F.elu(x)
        return self.out(x)

    # def inference(self, x, edge_index, edge_type, edge_attr, keywords):
    #     self.eval()
    #     with torch.no_grad():
    #         x = F.dropout(x, p=0.6, training=self.training)
    #         x = F.elu(self.conv1(x, edge_index, edge_type=edge_type, edge_attr=edge_attr))
    #         x = F.dropout(x, p=0.6, training=self.training)
    #         x = self.conv2(x, edge_index, edge_type=edge_type, edge_attr=edge_attr)
    #         x = self.out(x)

    #     # Mask for keywords
    #     keyword_mask = (x[:len(keywords)] != 0).any(dim=1)
    #     keyword_embeddings = x[keyword_mask]

    #     # Compute similarities to all act and obj nodes
    #     act_obj_embeddings = x[len(keywords):]
    #     similarities = torch.mm(keyword_embeddings, act_obj_embeddings.t())

    #     # Get top-5 related act and obj nodes
    #     top_indices = similarities.topk(5, dim=1).indices
    #     return top_indices


def get_K_AO(keyword_relation, out):
    # 创建一个布尔张量来标识 keyword 节点
    is_keyword = (keyword_relation != -1) & (keyword_relation != -99)
    is_keyword = is_keyword.any(dim=1)  # 对每一行应用any，如果行中包含非-1和非-99的元素，则为True

    # 输出 keyword 节点的 index
    keyword_indices = torch.nonzero(is_keyword, as_tuple=False).squeeze()
    # 获取 act/obj 节点的索引
    act_obj_indices = torch.nonzero(~is_keyword, as_tuple=False).squeeze()
    if act_obj_indices.dim()==0 and act_obj_indices.numel() == 1:
        print("1123")

    # 从out中选择 keyword 节点和 act/obj 节点的表示
    K = out[is_keyword]  # keyword 节点的表示
    AO = out[~is_keyword]  # act/obj 节点的表示

    return K, AO, keyword_indices, act_obj_indices


def get_uc_to_keyword(toUCText):
    # 将张量转换回列表
    to_UCText = toUCText.tolist()

    # 创建一个空字典来存储结果
    B = {}

    # 遍历 to_UCText 列表中的每个子列表
    for node_id, uc_list in enumerate(to_UCText):
        # 遍历子列表中的每个 UC id
        for uc_id in uc_list:
            # 如果 UC id 不是 -99，则将其添加到字典中
            if uc_id != -99:
                # 如果 UC id 已经在字典中，则添加当前节点 id 到其值列表中
                if uc_id in B:
                    B[uc_id].append(node_id)
                    # 否则，创建一个新列表，只包含当前节点 id
                else:
                    B[uc_id] = [node_id]
    return B


def get_ground_truth(UC_to_act_obj, R):
    uctext_act_obj_matrix_expect = torch.zeros((len(UC_to_act_obj), R.size(1)))
    # 填充 expect_matrix
    for uctext_idx, (uctext, act_obj_list) in enumerate(UC_to_act_obj.items()):
        for act_obj_id in act_obj_list:
            uctext_act_obj_matrix_expect[uctext_idx, act_obj_id] = 1

    return uctext_act_obj_matrix_expect

def get_predictions(UC_to_keyword, R, device):
    uctext_act_obj_matrix = torch.zeros((len(UC_to_keyword), R.size(1)))
    # 填充 UCText-act/obj 矩阵
    for uctext_idx, (uctext, keyword_indices) in enumerate(UC_to_keyword.items()):
        sum_vector = torch.zeros(R.size(1)).to(device)  # 初始化向量求和
        for keyword_idx in keyword_indices:
            sum_vector += R[keyword_idx]  # 对 keyword 行进行求和
        # 取平均并累加到新行向量中
        uctext_act_obj_matrix[uctext_idx] = sum_vector / len(keyword_indices)

    return uctext_act_obj_matrix

def get_uc_to_node(UC_to_all_node, keyword_index,act_obj_indices):
    UC_to_keyword = {}  # 从 UC_to_all_node 中提取的
    UC_to_act_obj = {}  # 从 UC_to_all_node 中提取的
    for uc in sorted(UC_to_all_node.keys()):
        # 5.1.1 找到每个uc对应节点中的keyword节点、act\obj节点
        keyword_list = set(UC_to_all_node[uc]).intersection(set(keyword_index.tolist()))
        keyword_list = sorted(keyword_list)  # 每个UC对应的keyword
        act_obj_list = [item for item in UC_to_all_node[uc] if item not in keyword_list]  # 每个UC对应的act\obj节点

        # 5.1.2 转变成 UCText - act/obj 的矩阵
        # 将UC_to_keyword中keyword的全局id转变为R中的局部id
        keyword_list_local = [idx for idx, val in enumerate(keyword_index.tolist()) if val in keyword_list]
        UC_to_keyword[uc] = keyword_list_local
        try:
            act_obj_list_local = [idx for idx, val in enumerate(act_obj_indices.tolist()) if val in act_obj_list]
        except:
            print("error_indices:")
            print(act_obj_indices)
            print("error.")
        UC_to_act_obj[uc] = act_obj_list_local
    return UC_to_keyword, UC_to_act_obj


def get_two_matrix(data, out, device):
    # 3、得到keyword的矩阵K，和act/obj的矩阵AO
    K, AO, keyword_index, act_obj_indices = get_K_AO(data.keyword_relation, out)

    # 4、计算R=K*AO的转置矩阵,并对R做sigmoid归一化
    R = torch.matmul(K, AO.t())
    R = torch.sigmoid(R) # SHAPE: [Keywords, ACT+OBJ]

    # 5、处理模型输出，和 ground truth
    # 5.1、获取 UCText - act/obj 的矩阵
    UC_to_all_node = get_uc_to_keyword(data.to_UCText)  # 每个UCText对应的所有点
    UC_to_keyword, UC_to_act_obj = get_uc_to_node(UC_to_all_node, keyword_index, act_obj_indices)

    # 创建一个空的张量来存储UCText-act/obj矩阵,即为预测值: uctext_act_obj_matrix
    uctext_act_obj_matrix = get_predictions(UC_to_keyword, R, device)

    # 5.2、获取ground truth，即UCText-act/obj的相应位置为1，其余为0的：uctext_act_obj_matrix_expect
    uctext_act_obj_matrix_expect = get_ground_truth(UC_to_act_obj, R)

    return uctext_act_obj_matrix, uctext_act_obj_matrix_expect



# 之前将问题定义为“推荐问题”用到的度量
metric_collection_recommend = MetricCollection([
    RetrievalMAP(top_k=10),
    RetrievalAUROC(top_k=10),
    RetrievalMRR(top_k=10),
    RetrievalRecall(top_k=10)
])

metric_collection = MetricCollection([
    # BinaryPrecision(),
    # BinaryRecall(),
    # BinaryF1Score(),
    BinaryAUROC(thresholds=None)
])