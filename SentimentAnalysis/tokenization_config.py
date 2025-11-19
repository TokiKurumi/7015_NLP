# tokenization_config.py
TOKENIZATION_CONFIG = {
    'word': {
        'name': 'Word-level',
        'description': '单词级别分词'
    },
    'char': {
        'name': 'Character-level',
        'description': '字符级别分词'
    },
    'subword': {
        'name': 'Subword-level',
        'description': '子词级别分词',
        'params': {
            'min_n': 2,
            'max_n': 4
        }
    }
}