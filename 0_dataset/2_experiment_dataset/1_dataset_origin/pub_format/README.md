pub dataset 中的 uctext 和 step 进行形式化处理（ernie3.5断句，gpt3.5形式化）。
尽量保持一句包含一对act-obj；去除多余副词。
任务位置在实验室电脑的 PE_based_public_dataset.py 的 task16.

pub_only_format_uctext.json 是只将uctext进行简单断句后的结果。
pub.json是将 step 进行形式化处理后的结果(本文件夹下的最终结果)。