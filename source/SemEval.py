import numpy as np
import pandas as pd


class SemEvalStruct(list):
    class Node(str):
        def __init__(node, aspect: str):
            node.aspect = aspect

        def __repr__(node):
            return str(node.aspect)

        def __str__(node):
            return str(node.aspect)

    def __init__(self, text: str):
        super().__init__()
        self.num_aspect = text.count("([")
        self.index_set = []
        self.caption, stream = text[:text.find('####')], text[text.find('####'):]
        self.extraction(stream)

    def extraction(self, stream: str):
        bracket_pos = 0
        for ind in range(0, self.num_aspect):
            bracket_pos = stream.find("([", bracket_pos+4)
            aspect_index = list(map(int, stream[stream.find('[', bracket_pos):stream.find(']', bracket_pos)].replace("[", "").replace("(", "").replace("]", "").split(", ")))
            sam_eval_aspect = [word for word in self.caption.split(" ") if self.caption.split(" ").index(word) in aspect_index]

            self.append(self.Node(" ".join(sam_eval_aspect)))
            self.index_set.append(aspect_index)
        # self.index_set = [item for element in self.index_set for item in element]


def read_sem_eval(load_path: str) -> pd.DataFrame:
    file = open(load_path)
    array = [[], [], []]
    for line in file.readlines():
        sam_eval_struct = SemEvalStruct(line)

        array[0].append(sam_eval_struct.caption)
        array[1].append(sam_eval_struct.index_set)
        array[2].append(sam_eval_struct)

    sem_eval_dataframe = pd.DataFrame(np.array(array, dtype='object').transpose(),
                                      columns=["caption", "aspect_index", "sam_eval_aspect"])
    return sem_eval_dataframe