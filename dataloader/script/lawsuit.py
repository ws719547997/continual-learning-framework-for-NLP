import json

from datasets import Value, ClassLabel, Sequence
import datasets


_LAWSUIT_CITATION = """\

"""

_LAWSUIT_DESCRIPTION = """\
GLUE, the General Language Understanding Evaluation benchmark
(https://gluebenchmark.com/) is a collection of resources for training,
evaluating, and analyzing natural language understanding systems.
"""

class LAWSUITConfig(datasets.BuilderConfig):

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        data_dir,
        citation,
        url,
        label_classes=None,
        process_label=lambda x: x,
        **kwargs,
    ):
        super(LAWSUITConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.data_dir = data_dir
        self.citation = citation
        self.url = url
        self.process_label = process_label

class LAWSUIT(datasets.GeneratorBasedBuilder):
    domain_list = []

    BUILDER_CONFIGS  = [
        LAWSUITConfig(name=domain_name,
                    description= f'comments of JD {domain_name}.', 
                    text_features={'id':'id',
                                   'content':'content',
                                   'qa_pairs':'qa_pairs'
                                   },
                    citation="",
                    data_dir= "",
                    data_url = r"https://huggingface.co/datasets/kuroneko5943/lawsuit/resolve/main/",
                    url='http://contest.aicubes.cn/')
        for domain_name in domain_list
    ]

    def _info(self):
        features = {
            'id':Value(dtype='int32', id=None),
            'content':Value(dtype='string', id=None),
            'qa_pairs':Sequence(feature={
                'question': Value(dtype='string', id=None),
                'blocks':Sequence(feature={
                    'text':Value(dtype='string', id=None),
                    'start':Value(dtype='int32', id=None)
                },
                    id=None,
                    length=-1)
            },
                id=None,
                length=-1)
        }

        return datasets.DatasetInfo(
            description=_LAWSUIT_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _LAWSUIT_CITATION,
        )

    def _split_generators(self, dl_manager):

        test_file = rf'{self.config.data_url}{self.config.name}//test.json'
        dev_file = rf'{self.config.data_url}{self.config.name}//dev.json'
        train_file = rf'{self.config.data_url}{self.config.name}//train.json'

        return [datasets.SplitGenerator(name=datasets.Split.TEST,
                                        gen_kwargs={
                                            "data_file": dl_manager.download(test_file),
                                            "split": "test",
                                        },), 
                datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                        gen_kwargs={
                                            "data_file": dl_manager.download(dev_file),
                                            "split": "dev",
                                        },), 
                datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                        gen_kwargs={
                                            "data_file": dl_manager.download(train_file),
                                            "split": "train",
                                        },)]

    def _generate_examples(self, data_file, split):
        with open(data_file, 'r', encoding='utf-8') as f:
            for id, line in enumerate(f):
                json_line = json.loads(line)
                '''
                在标注数据中存在多组问句的情况，但是我们通常认为多组问句应当属于不同的训练任务，因此在语料中只取第一组问题；
                如果有多个问题组会被分成多个语料。
                同时一个问题可能包含多个答案，每个答案包含文本和起始位置，在查看语料时会更加直观，同时避免重复。
                '''
                data = {}
                data['id'] = id
                data['content'] = json_line['contents'][0]['content']
                question_list = []
                for question in data['annotations'][0]['qa_pairs']:
                    answer_list = []
                    for ans in question['blocks']:
                        answer_list.append({
                            'text':ans['text'],
                            'start':ans['start']
                        })
                    question_list.append({
                        'question':question['question_name'],
                        'blocks':answer_list
                    })
                data['qa_pairs'] = question_list
                # id, sample
                yield data['id'], data
