import sqlite3
from contextlib import closing
from pathlib import Path
import csv
from typing import List, Tuple, Callable, Iterable, Union, Dict, Optional
import re

import nltk
from nltk import word_tokenize
import pandas as pd

TAG_OUTER = 'O'

NE5_dataset = Path('/data/NER/VectorX/NE5_train')
# NE5_dataset = Path('/data/NER/Academic Datasets/NE5/Collection5')


_POPULAR_SHORTENINGS = re.compile(r'\b(ул|тел|корп|зам|им|руб|коп|адм|обл|тыс|эт|сан|барр)\.$', re.IGNORECASE)
_HAS_DOT_INSIDE = re.compile(r'[\w]+\.[\w]+\.$', re.IGNORECASE)
_INITIALS = re.compile(r'\b[А-Я]{1,4}\.$')
_ONLY_CONSONANTS = re.compile(r'\b[бвгджзйклмнпрстфхцчшщ]{1,4}\.$', re.IGNORECASE)
def loose_sent_tokenize(text: str):
    sentences = []
    sents = nltk.sent_tokenize(text)
    si = 0
    processed_index = 0
    sent_start = 0
    while si < len(sents):
        s = sents[si]
        span_start = text[processed_index:].index(s) + processed_index
        span_end = span_start + len(s)
        processed_index += len(s)

        if _POPULAR_SHORTENINGS.search(s):
            pass
        elif _ONLY_CONSONANTS.search(s):
            pass
        elif _INITIALS.search(s):
            pass
        elif _HAS_DOT_INSIDE.search(s):
            pass
        else:
            if not text[sent_start: span_end].strip():
                print('empty!')
            sentences.append(text[sent_start: span_end].strip())
            sent_start = span_end
            processed_index = span_end

        si += 1


    if sent_start != len(text):
        if text[sent_start:].strip():
            sentences.append(text[sent_start:].strip())
    return sentences

text = '''Что это такое?
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 Не пропусти ни одной новости – подписывайся на новый сервис от 24 канала
 Что это такое?
 
 Оперативно о самом главном – сервис "Важно!" от 24 канала
 Что это такое?
 
 Узнавай первым о самом главном
 Что это такое?
 
 Мгновенные сообщения о главных новостях мира
 Что это такое?
 
 Будь в курсе событий в стране и мире – подписывайся на сервис "Важно!"
 Что это такое?
 
 
 Подписаться
 
 
 

 
 
 
 
 
 
 Читай главные новости мира в Viber
 Что это такое?
 
 24 канал теперь в Viber. Подписывайся на наш паблик-аккаунт!
 Что это такое?
 
 Все новости всегда под рукой. Чат 24 канала в Viber
 Что это такое?
 
 Усі новини завжди під рукою. Чат 24 каналу у Viber
 Что это такое?
 
 Удобная доставка новостей в Viber
 Что это такое?
 
 Читай новости от 24 канала в Viber
 Что это такое?
 
 Лаконично и оперативно – новости 24 канала в Viber
 Что это такое?
 
 
 Подписаться
 
 
 

 
 
 
 
 
 

 Самые главные новости в твоем Мессенджере. Присоединяйся!
 Что это такое?
 
 Удобная доставка новостей от бота 24 канала
 Что это такое?
 
 "Привет! Для тебя новость". Бот 24 канала в мессенджер
 Что это такое?
 
 Твой вопрос – наша новость. Бот 24 канала
 Что это такое?
 
 Испытай возможности новостного бота 24 канала
 Что это такое?
 
 Есть ли новости, о которых он не знает? Бот от 24 канала
 Что это такое?
 

 
 Подписаться
 
 
 

 
 
 
 
 
 

 Самые главные новости в твоем Мессенджере. Присоединяйся!
 Что это такое?
 
 Удобная доставка новостей от бота 24 канала
 Что это такое?
 
 "Привет! Для тебя новость". Бот 24 канала в мессенджер
 Что это такое?
 
 Твой вопрос – наша новость. Бот 24 канала
 Что это такое?
 
 Испытай возможности новостного бота 24 канала
 Что это такое?
 
 Есть ли новости, о которых он не знает? Бот от 24 канала
 Что это такое?
 

 
 Подписаться
 
 
 








 На железнодорожном вокзале в Киеве известная украинская писательница Лариса Ницой стала свидетелем одной довольно интересной сцены, когда маленький мальчик удивленно отреагировал на железнодорожное сообщение между Украиной и страной-агрессором Россией.
 
Об этом она написала на своей странице в Facebook. 

"Киев. Железнодорожный вокзал. К центральному залу заходит папа за руку с небольшим сыном. Кого-то встречают. Ищут взглядом табло. По всему видно, сынок уже умеет читать. Прочитав верхние две строчки, ребенок округляет глаза, его лицо удлиняется, он подается весь вперед, вырывает руку с папиной руки и показывает на табло: "А это что-о-о-о-о-о-о ?! – вскрикивает он. – У нас на Москву позда-а-а-а ходят?!!" – отметила писательница. 
Читайте также: Политик объяснил, почему нельзя прекращать железнодорожное сообщение с Россией 

По ее словам, к такой реакции ребенка не остались равнодушными посетители вокзала. 

"На него оглядывается полвокзала. Немая сцена. Занавес", – добавила Ницой. 
 
 
 
Напомним, в конце мая российские СМИ сообщили, что Украина планирует окончательно приостановить железнодорожное сообщение с Россией. Однако министр инфрастурктуры Владимир Омелян впоследствии заметил, что в Кабмине пока такой вопрос не рассматривался.
 



 
 Теги:
 Новости Украины 
 Новости политики 
 Несовершеннолетние 
 Дети 
 Укрзализниця 
 железная дорога 
 Новости России 
 Украина – Россия'''

print(loose_sent_tokenize(text))


assert len(loose_sent_tokenize('купил за 5 руб. и остался доволен.')) == 1
assert len(loose_sent_tokenize('Я ему сказал и т.к. он не послушался за 500р.')) == 1
assert len(loose_sent_tokenize('Ура. Ура. 500р.')) == 3
assert len(loose_sent_tokenize('И.П. Павлов.')) == 1
assert len(loose_sent_tokenize('И. П. Павлов.')) == 1
assert len(loose_sent_tokenize('Я ему сказал: "Чтобы ничего не трогал." Но он не послушался.')) == 2
assert len(loose_sent_tokenize('Нефть за $27/барр. не снится.')) == 1


def get_charspans(text: str, tokenizer_or_words: Union[Callable[[str], List[str]], List[str]],
                  subs_map: Optional[Dict[str, str]]=None) -> List[Tuple[str, int, int]]:
    if subs_map is None:
        subs_map = {}

    wi = 0
    words = tokenizer_or_words(text) if callable(tokenizer_or_words) else tokenizer_or_words
    word_origs = [subs_map.get(w, w) for w in words]
    words_span = []

    i = 0
    while i < len(text) and wi < len(words):
        if text[i:].startswith(word_origs[wi]):
            words_span.append((words[wi], i, i+len(word_origs[wi])))
            i += len(word_origs[wi])
            wi += 1
            continue
        i += 1

    return words_span


t = 'some \n \ttext\n'
for st, ss, se in get_charspans(t, word_tokenize):
    assert st == t[ss: se]

for st, ss, se in get_charspans(t, ['some', 'text']):
    assert st == t[ss: se]


def split_sentences_preserve_tags(text: str,
                                  tags_with_spans: Iterable[Tuple[str, int, int, str]],
                                  sentence_tokenizer: Callable[[str], Iterable[str]]):
    tags_with_spans = sorted(tags_with_spans, key=lambda x: x[1])  # excess
    sents = sentence_tokenizer(text)
    if min(len(s) for s in sents) == 0:
        print('123')
    sents_span = get_charspans(text, sents)

    per_sentence_tags = []
    for orig_s, s_start, s_end in sents_span:
        sent_tags = []
        s = orig_s.strip()
        offset = orig_s.index(s)
        s_start = s_start + offset
        s_end = s_start + len(s)

        for tag, t_start, t_end, value in tags_with_spans:
            if s_start <= t_start <= s_end:
                assert s_start <= t_end <= s_end, 'Entity is in two sentences'
                assert s[t_start - s_start: t_end - s_start] == value, 'Tag value does not equal to sent spans'
                sent_tags.append((tag, t_start - s_start, t_end - s_start, value))
        per_sentence_tags.append((s, sent_tags))
    return per_sentence_tags


def charspans_to_bio(word_spans: List[Tuple[str, int, int]], tags: List[Tuple[int, int, str]]) -> List[Tuple[str, str]]:
    """supposed that tags do not overlap

    returns list of pairs of word and its bio-tag"""

    if not tags:
        return [(w[0], TAG_OUTER) for w in word_spans]

    tags = sorted(tags, key=lambda x: x[0])
    for t1, t2 in zip(tags[:-1], tags[1:]):
        assert t1[1] <= t2[0], 'Tags must not overlap'

    bio = []
    is_tag_started = False
    wi = 0
    ti = 0

    if not tags:
        tags = [('', word_spans[-1][-1], word_spans[-1][-1])]

    while wi < len(word_spans) and ti < len(tags):
        w, w_start, w_end = word_spans[wi]
        t_start, t_end, tag = tags[ti]
        if t_start > w_end:
            bio.append((w, TAG_OUTER))
            wi += 1
        elif t_start <= w_start < t_end or t_start < w_end < t_end:
            prefix = 'I-' if is_tag_started == tag else 'B-'
            bio.append((w, prefix + tag))
            wi += 1
            is_tag_started = True
        else:
            ti += 1
            is_tag_started = False

    while wi < len(word_spans):
        w, w_start, w_end = word_spans[wi]
        bio.append((w, TAG_OUTER))
        wi += 1

    assert len(bio) == len(word_spans), bio
    return bio

_test_cs = charspans_to_bio([('a', 0, 1), ('bb', 2, 4), ('cc', 5, 7)], [(2, 4, 'TEST'), (5, 7, 'TEST')])
assert _test_cs == [('a', TAG_OUTER), ('bb', 'B-TEST'), ('cc', 'B-TEST')], _test_cs


conn = sqlite3.connect(':memory:')

with closing(conn.cursor()) as c:
    c.execute('''CREATE TABLE sentences (article_id INT,
                                         order_in_article INT,
                                         dataset TEXT, 
                                         value TEXT,
                                         annotated INT)''')
    c.execute('''CREATE TABLE tags (sentence_id INT, 
                                    start_index INT, 
                                    end_index INT, 
                                    tag TEXT)''')
    conn.commit()
    print('Created empty database in memory')


def _add_sentence(c: sqlite3.Cursor, article_id: str, order_in_article: int, sent_text: str, dataset_name: str, tags: Optional[list]):
    is_annotated = int(tags is not None)
    r = c.execute('''INSERT INTO sentences(article_id, order_in_article, dataset, value, annotated) 
                     VALUES(?,?,?,?,?)''', (article_id, order_in_article, dataset_name, sent_text, is_annotated))
    s_id = r.lastrowid
    if tags:
        for st, ss, se, sv in tags:
            c.execute('INSERT INTO tags(sentence_id, start_index, end_index, tag) VALUES (?, ?, ?, ?)',
                      (s_id, ss, se, st))
    return s_id


def slurp_unsupervised_data(text: str,
                            text_id: Optional[str]=None,
                            dataset_name: str='',
                            sentence_tokenizer: Optional[Callable[[str], str]]=None):
    if sentence_tokenizer is None:
        sentence_tokenizer = loose_sent_tokenize

    with closing(conn.cursor()) as c:
        for order, s in enumerate(sentence_tokenizer(text)):
            _add_sentence(c, text_id, order, s.strip(), dataset_name, None)


def slurp_NE5_annotated_data(folder_path, dataset_name: str = ''):
    NE5_dataset = Path(folder_path)
    all_texts = []
    with closing(conn.cursor()) as c:
        for ann_fn in NE5_dataset.glob('*.ann'):
            #     ann_fn = Path('/data/NER/VectorX/NE5_train/82180290.ann')
            article_id = ann_fn.name[:-len('.ann')]
            text_fn = Path(str(ann_fn)[:-len('.ann')] + '.txt')

            text = text_fn.read_text()
            all_texts.append({'article_id': article_id, 'text': text})

            annotations = []
            with ann_fn.open() as f:
                cr = csv.reader(f, delimiter='\t', quotechar='|')
                for row in cr:
                    assert len(row) == 3, row
                    _, tag_info, tag_value = row
                    tag, start_index, end_index = tag_info.split()
                    annotations.append((tag, int(start_index), int(end_index), tag_value))

            try:
                st = split_sentences_preserve_tags(text, annotations, loose_sent_tokenize)

                for order, (s, stags) in enumerate(st):
                    _add_sentence(c, article_id, order, s, dataset_name, stags)
                    # r = c.execute('''INSERT INTO sentences(article_id, order_in_article, dataset, value, annotated)
                    #                  VALUES(?,?,?,?,?)''', (article_id, order, dataset_name, s, 1))
                    # s_id = r.lastrowid
                    # for st, ss, se, sv in stags:
                    #     c.execute('INSERT INTO tags(sentence_id, start_index, end_index, tag) VALUES (?, ?, ?, ?)',
                    #               (s_id, ss, se, st))
                conn.commit()
            except Exception as ex:
                print(f'Exception "{ex}" for file {text_fn}')
                # print(sent_tokenize(text))
                # print('='*80)


def get_unsupervised_data(dataset_name=''):
    with closing(conn.cursor()) as c:
        r = c.execute('''SELECT oid, 
                                article_id, 
                                value
                         FROM sentences
                         WHERE annotated = 0 AND dataset=?''', (dataset_name,))
        ret = []
        for rowid, article_id, text in r.fetchall():
            ret.append({'rowid': rowid, 'article_id': article_id, 'sentence': text})
    return ret


def get_supervised_data(dataset_name='',
                        tokenizer: Optional[Callable[[str], str]]=None,
                        substitution_map: Optional[Dict[str, str]]=None):
    if tokenizer is None:
        tokenizer = word_tokenize
    if substitution_map is None:
        substitution_map = {"''": '"', '``': '"'}

    sents = []
    with closing(conn.cursor()) as c:
        r = c.execute('''SELECT sentences.oid, 
                                sentences.article_id, 
                                sentences.value, 
                                tags.start_index, 
                                tags.end_index, 
                                tags.tag
                         FROM sentences LEFT OUTER JOIN tags ON sentences.oid = tags.sentence_id 
                         WHERE sentences.annotated = 1 and dataset=?''', (dataset_name,))
        d = r.fetchall()

        data = pd.DataFrame(d, columns='sentences.oid sentences.article_id sentences.value tags.start_index tags.end_index tags.tag'.split())


        for sid, d in data.groupby('sentences.oid'):
            s = d['sentences.value'].iloc[0]
            tags = []
            for _, row in d.iterrows():
                if not pd.isna(row['tags.start_index']):
                    tags.append((int(row['tags.start_index']), int(row['tags.end_index']), row['tags.tag']))

            word_spans = get_charspans(s, tokenizer, substitution_map)
            words, bio_tags = zip(*charspans_to_bio(word_spans, tags))
            sents.append((words, bio_tags))


    return sents


def _replace_nan(ax: list, with_value):
    return [with_value if a is None else a for a in ax]


def get_data_as_pandas():
    with closing(conn.cursor()) as c:
        r = c.execute('''SELECT sentences.oid, 
                                sentences.article_id, 
                                sentences.order_in_article,
                                sentences.value, 
                                sentences.dataset,
                                sentences.annotated,
                                tags.start_index, 
                                tags.end_index, 
                                tags.tag
                         FROM sentences LEFT OUTER JOIN tags ON sentences.oid = tags.sentence_id''')
        oid, article_id, order_in_article, value, dataset, annotated, start, end, tag = zip(*r.fetchall())
        cols = [
            pd.Series(oid, name='oid', dtype=int),
            pd.Series(article_id, name='article_id'),
            pd.Series(order_in_article, name='order_in_article'),
            pd.Series(value, name='value'),
            pd.Series(dataset, name='dataset'),
            pd.Series(annotated, name='annotated', dtype=int),
            pd.Series(_replace_nan(start, -1), name='start', dtype=int),
            pd.Series(_replace_nan(end, -1), name='end', dtype=int),
            pd.Series(tag, name='tag')]
        return pd.concat(cols, axis=1)


def word_indexi_to_spans(text: str, word_spans: List[Tuple[str, int, int]], subs_map=None):
    if subs_map is None:
        subs_map = {}

    wi = 0

    words = [ws[0] for ws in word_spans]
    word_origs = [subs_map.get(w, w) for w in words]
    words_span = []

    i = 0
    while i < len(text) and wi < len(words):
        if text[i:].startswith(word_origs[wi]):
            words_span.append((words[wi], i, i+len(word_origs[wi])))
            i += len(word_origs[wi])
            wi += 1
            continue
        i += 1

    return words_span


def set_annotation(sentence_rowid: int,
                   tokenization: Union[Callable[[str], List[str]], List[str]],
                   selections: List[Tuple[str, int, int]]):
    with closing(conn.cursor()) as c:
        c.execute('''SELECT value, annotated FROM sentences WHERE oid=?''', sentence_rowid)
        orig_sent, is_annotated = c.fetchone()

        assert not is_annotated, f'Error: sentence {sentence_rowid} has already been annotated'

        words_spans = get_charspans(orig_sent, tokenization)

        spans_to_insert = []
        for tag, w_start, w_end_incl in selections:
            spans_to_insert.append((sentence_rowid, words_spans[w_start][1], words_spans[w_end_incl][2], tag))

        c.execute('''UPDATE sentences SET annotated=1 WHERE oid=?''', sentence_rowid)
        c.executemany('''INSERT INTO tags (sentence_id, start_index, end_index, tag) VALUES (?, ?, ?, ?)''',
                      spans_to_insert)

        conn.commit()


if __name__ == '__main__':
    slurp_NE5_annotated_data('/data/NER/VectorX/NE5_train', 'train')
    train = get_supervised_data('train')
    words, tags = zip(*train)
    for ws in words:
        if 'Древлянке' in ws:
            print(ws)


