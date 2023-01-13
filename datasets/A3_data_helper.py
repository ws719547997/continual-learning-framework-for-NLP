import json
import os
import time
import zipfile
import random
from typing import Union


def make_dir(path: str) -> None:
    """
    检查输入路径是否存在，如果有，则不操作，如果没有，则创建。
    :param path: 输入的路径
    :return: None
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'创建路径：{path}。')
    else:
        print(f'路径{path}已经存在。')


def extract_zip(file_path: str, output_path: str) -> bool:
    """
    从平台导出的zip语料文件中读取文件，返回content.json和meta.json的字典对象
    :param file_path:压缩文件路径
    :param output_path:解压缩路径
    :return: [content_dict, meta_dict]
    """
    if not zipfile.is_zipfile(file_path):
        print(f'{file_path}不是zip文件')
        return False
    else:
        with zipfile.ZipFile(file_path, 'r') as zfile:
            zfile.extractall(output_path)
        return True


def write_zip(zip_path: str, input_file: Union[str, list], save_name: Union[str, list]) -> None:
    with zipfile.ZipFile(zip_path, 'w') as zfile:
        if isinstance(input_file, list):
            for file_dir, file_name in zip(input_file, save_name):
                zfile.write(file_dir, file_name)
        else:
            zfile.write(input_file, save_name)


def get_filelist(path: str) -> list:
    for root, dirs, files in os.walk(path):
        print(f"files: {files}")
    return files


def load_content(input_dir: str) -> str:
    """
    按行返回文件的内容
    :param input_dir:存储content.json文件的目录
    :yield:返回语料中一行
    """
    with open(input_dir + 'content.json', 'r', encoding='utf-8') as f:
        for line in f:
            yield line


def save_as_json(input_list: list, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as f:
        for i in input_list:
            f.write(json.dumps(i, ensure_ascii=False) + '\n')


class A3Dataset:
    def __init__(self,
                 dataset_name: str,
                 zipfile_dir: Union[str, list],
                 output_dir: str,
                 split_dict: dict = None,
                 seed: int = 511):

        self.dataset_name: str = dataset_name
        self.zipfile_dir: str = zipfile_dir
        self.output_dir: str = output_dir
        self.split_dict: dict = split_dict
        self.seed: int = seed
        self.status_map: dict = {0: 'unlabel', 1: 'labeled', 3: 'discard'}
        self.splited_data: dict = None
        self.meta: dict = None

        make_dir(self.output_dir)
        if extract_zip(self.zipfile_dir, f'{self.output_dir}'):
            self.meta = self._get_meta()
            self.contents: list = self._get_content()
            self.question_dict = self._get_question_dict()
            # self.content_type = self._get_content_type()

    def _get_meta(self) -> Union[dict, None]:
        """
        从meta.json中读取配置信息并转化为字典。
        :return: dict或者None
        """
        if 'meta.json' in get_filelist(self.output_dir):
            with open(f'{self.output_dir}meta.json', 'r', encoding='utf-8') as f:
                # meta文件有两种形式，一种是一行json，一种是按格式输出。
                meta = json.load(f)
            return meta
        else:
            return None

    def _get_question_dict(self) -> dict:
        """
        在老的数据集中，content.json中没有问题名称，仅有问题id，因此需要meta文件中问题id和名称的映射。
        :return:
        """
        if self.meta is None:
            print('zip中不包含meta.json文件，或输入路径不是zip文件。')
        else:
            question_dict = {}
            # 在这里遍历时，如果有多组问题会都被加进来，但是从模型训练的角度来看，多组问题就应该被划分成多个不同的数据集。
            for q_group in self.meta['question_groups']:
                for q in q_group['questions']:
                    # key是不会重复的，name有可能重复。
                    question_dict.update({q['key']: q['name']})
            return question_dict

    def _get_content_type(self) -> str:
        """
        语料的类型有text和pdf两种，目前部分操作仅能考虑到text的情况。
        :return:
        """
        if self.meta is None:
            print('zip中不包含meta.json文件，或输入路径不是zip文件。')
        else:
            return self.meta['content_coms'][0]['tag']

    def _get_count_info(self) -> dict:
        """
        读取meta文件中已标注、未标注、拉黑数据分别的数量
        :return: {'数据标注状态'：数量，}
        """

        if self.meta is None:
            print('zip中不包含meta.json文件，或输入路径不是zip文件。')
        else:
            count_info_dict = {}

            for item in self.meta['count_info']:
                count_info_dict.update({self.status_map[item['status']]: item['count']})
        return count_info_dict

    def _get_content(self) -> list:
        return [json.loads(i) for i in load_content(f'{self.output_dir}')]

    def select_by_status(self, status_list: list) -> None:
        temp_list = []
        for item in self.contents:
            if item['annot_status'] in status_list:
                temp_list.append(item)
        self.contents = temp_list

    def select_by_user(self, user_name_list: list) -> None:
        temp_list = []
        for item in self.contents:
            if item['annot_user'] in user_name_list:
                temp_list.append(item)
        self.contents = temp_list

    def select_by_time(self, start=None, end=None):
        temp_list = []
        for item in self.contents:
            # 设定了起始和结束事件范围：（start,end）
            # 如果没有设置开始或者结束范围的时候，我们可以把它设置为一个很大的值以囊括当前的数据。
            start = start if start is not None else '2000-01-01 00:00:00'
            end = end if end is not None else '2030-01-01 00:00:00'

            if int(time.mktime(time.strptime(start, '%Y-%m-%d %H:%M:%S'))) < \
                    int(time.mktime(time.strptime(item['annot_time'], '%Y-%m-%d %H:%M:%S'))) < \
                    int(time.mktime(time.strptime(end, '%Y-%m-%d %H:%M:%S'))):
                temp_list.append(item)
        self.contents = temp_list

    def split_data(self, split_dict: dict = None) -> None:
        '''
        把{{'train': [], 'dev': [], 'test': [data,data,...]}} 存在self.splited_data里面。
        :param split_dict: like:{'train': 0.7, 'dev': 0.1, 'test': 0.2}
        '''
        if split_dict is not None:
            self.split_dict = split_dict

        if self.contents is not None:
            ratio = 0.0
            length = len(self.contents)
            data_dict = {}
            random.seed(self.seed)
            random.shuffle(self.contents)
            for k, v in self.split_dict.items():
                data_dict.update({k: self.contents[int(ratio * length):int((ratio + v) * length)]})
                ratio += v
            self.splited_data = data_dict

    def save_split_data(self, splited_data: dict = None) -> None:
        """
        :param splited_data: 划分好的数据字典
        :return:
        """
        if splited_data is not None:
            self.splited_data = splited_data
        for k, v in self.splited_data.items():
            with open(f'{self.output_dir}\\{k}.json', 'w', encoding='utf-8') as f:
                for line in v:
                    f.write(json.dumps(line, ensure_ascii=False) + '\n')
            print(f'{k} data saved! at {self.output_dir}.')

    def set_question_name(self) -> None:
        """
        在老版本的数据集中，没有question_name字段，
        因此需要从meta中找到question_name和question_key的映射关系，
        把question_name加入到语料中。
        :return: None
        """
        if self.contents is not None:
            for annot in self.contents:
                for q_group in annot['annotations']:
                    for qa_pair in q_group['qa_pairs']:
                        qa_pair['question_name'] = self.question_dict[qa_pair['question_key']]
        else:
            print(f'请先将语料存入self.contents。')

    def save_contents_zip(self):
        self.save_contents_json()
        write_zip(zip_path=f'{self.output_dir}{self.dataset_name}.zip',
                  input_file=[f'{self.output_dir}{file_name}' for file_name in ['meta.json', 'processed_content.json']],
                  save_name=['meta.json', 'contnet.json'])

    def save_contents_json(self):
        save_as_json(self.contents, self.output_dir + 'processed_content.json')


if __name__ == '__main__':
    zipfile_dir = r'D:\Personal\Work\Corpus\基金经理简历格式化_工作经历.zip'
    dataset_name = zipfile_dir.split('\\')[-1].split('.')[0]
    output_dir = rf'D:\Personal\Work\Corpus\{dataset_name}\\'
    split_dict = {'train': 0.7, 'dev': 0.1, 'test': 0.2}

    dataset = A3Dataset(dataset_name=dataset_name,
                        zipfile_dir=zipfile_dir,
                        output_dir=output_dir,
                        split_dict=split_dict)

    # 筛选标注内容
    dataset.select_by_status([1])

    # 旧数据集，增加问题名称
    dataset.set_question_name()

    # 划分数据集
    dataset.split_data()

    # 保存划分过的数据集
    dataset.save_split_data()

    # 保存筛选过的数据集为zip
    # dataset.save_contents_zip()
    print('done.')
