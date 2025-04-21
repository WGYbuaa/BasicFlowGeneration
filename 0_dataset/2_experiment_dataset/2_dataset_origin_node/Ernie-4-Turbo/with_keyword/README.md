with_keyword中的文件，相比于Ernie-4-Turbo中同名的文件，
差别在于增加了从uctext、ucpath中提取的keyword，
用于后续制作.pt文件。

该文件中包含 ：index，step（原文）；act(从原文step中提取的act)；obj(从原文step中提取的obj)；
dataset（该uc从属于哪个dataset）；ucText（该uc的uctext）；ucPath(该uc原文件存放的相对路径)；
key_act和key_obj是从uctext中提取的node；key_path是从ucPath中提取的node。


该文件夹下的pub文件，是没有按照step存储的。按照step存储的最新版本在4_dataset_pred_node\ERNIE_4_Turbo_8k\with_tp\3rd_round\Ernie_pub.json