import backoff
from bidict import bidict
import openai
from openai.error import RateLimitError

label2desc = bidict({'DISEASE': 'Disease',
                     'CHEMICAL': 'Chemical',
                     'GENE': 'Gene',
                     'PER': 'Person',
                     'ORG': 'Organization',
                     'MISC': 'Miscellaneous',
                     'LOC': 'Location',
                     'GPE': 'Geopolitical',
                     'PERSON': 'People',
                     'NORP': 'Nationality',
                     'WORK_OF_ART': 'Art',
                     'FAC': 'Facility',
                     'LAW': 'Law',
                     'PRODUCT': 'Product',
                     "LANGUAGE": 'Language',
                     "EVENT": 'Event',
                     'VEH': 'Vehicle',
                     'WEA': 'Weapon'})

# TODO: add your API key here
api_keys = []


@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_gpt_backoff(**kwargs):
    api_idx = 0
    timeout_ = True
    while timeout_:
        openai.api_key = api_keys[api_idx]
        try:
            response = openai.Completion.create(**kwargs)
            timeout_ = False
        except Exception:
            timeout_ = True
            api_idx += 1
            if api_idx == len(api_keys):
                raise
    return response


def format_en_entities(pred_text, raw_words, label_list):
    raw_words = [word.lower() for word in raw_words]
    raw_text = ' '.join(raw_words)
    pred_text = pred_text.lower().strip()
    pred_entities = []
    format_negative = 0
    span_negative = 0

    if pred_text == 'none' or pred_text == 'none.' or pred_text == 'no' or pred_text == 'no.':
        return pred_entities, 0, 0, 0

    if len(label_list) == 1:
        if ':' in pred_text:
            pred_text = pred_text.split(':')[1].strip()
        elif ' - ' in pred_text:
            pred_text = pred_text.split(' - ')[1].strip()
        pred_labels = pred_text.split(',')
        pred_dicts = {label_list[0]: pred_labels}
    else:
        label_list = [label.lower() for label in label_list]
        # split event type and event span
        if '\n' in pred_text:
            pred_labels = pred_text.split('\n')
        elif ';' in pred_text:
            pred_labels = pred_text.split(';')
        else:
            for t in list(label2desc.values()):
                if t.lower() == pred_text and t.lower() in label_list:
                    ent_type = label2desc.inverse[t]
                    pred_entities.append((ent_type.lower(), (0, len(raw_words))))
                    return pred_entities, 0, 0, 0
            pred_labels = [pred_text]

        pred_dicts = {}
        for p_label in pred_labels:
            p_label = p_label.strip()
            if ":" in p_label:
                ent_type = p_label.split(':')[0].strip()
                ent_spans = p_label.split(':')[1]
            elif ' - ' in p_label:
                ent_type = p_label.split(' - ')[0].strip()
                ent_spans = p_label.split(' - ')[1]
            elif p_label in label2desc.values():
                ent_type = p_label
                ent_spans = raw_text.strip()
            else:
                format_negative += 1
                continue
            for t in list(label2desc.values()):
                if t.lower() in ent_type and t.lower() in label_list:
                    ent_type = label2desc.inverse[t]
                    pred_dicts[ent_type] = ent_spans.split(',')
                    break
            if ent_type not in list(label2desc.keys()):
                span_negative += 1
                continue

    # split event spans with same event type
    for ent_type in pred_dicts:
        ent_spans = pred_dicts[ent_type]
        for ent_span in ent_spans:
            ent_span = ent_span.strip()
            if len(ent_span) == 0:
                span_negative += 0
                continue
            if ent_span == 'none':
                continue
            if ent_span[-1] == '.' or ent_span[-1] == ';':
                ent_span = ent_span[:-1]
            if '-' in ent_span and ' - ' not in ent_span and ' - ' in raw_text:
                ent_span = ' - '.join(ent_span.split('-'))
            if '(' in ent_span and ')' in ent_span and ' ( ' in raw_text:
                ent_span = ' ( '.join(ent_span.split('('))
                ent_span = ' ) '.join(ent_span.split(')'))
            if ent_span not in raw_text:
                span_negative += 1
                continue
            ent_span = ent_span.split()
            if len(ent_span) == 0:
                span_negative += 1
                continue
            try:
                starts = [s_idx for s_idx, word in enumerate(raw_words) if word == ent_span[0]]
                assert len(starts) > 0
                for start in starts:
                    end = start + len(ent_span)
                    if end - 1 >= len(raw_words) or raw_words[end - 1] != ent_span[-1]:
                        span_negative += 1
                        continue
                    pred_entities.append((ent_type.lower(), (start, end)))
            except (ValueError, AssertionError):
                span_negative += 1

    if format_negative > 0 or span_negative > 0:
        sent_invalid = 1
    else:
        sent_invalid = 0

    return pred_entities, sent_invalid, format_negative, span_negative


def format_cn_entities(pred_text, raw_words, label_list):
    raw_text = ''.join(raw_words)
    pred_text = pred_text.lower().strip()
    pred_entities = []
    format_negative = 0
    span_negative = 0

    if pred_text == '无' or pred_text == '无。':
        return pred_entities, 0, 0, 0

    label_list = [label.lower() for label in label_list]
    # split event type and event span
    if '\n' in pred_text:
        pred_labels = pred_text.split('\n')
    elif ';' in pred_text:
        pred_labels = pred_text.split(';')
    elif '；' in pred_text:
        pred_labels = pred_text.split('；')
    elif '：' in pred_text or ':' in pred_text:
        pred_labels = [pred_text]
    else:
        for t in list(label2desc.values()):
            if t.lower() == pred_text and t.lower() in label_list:
                ent_type = label2desc.inverse[t]
                pred_entities.append((ent_type.lower(), (0, len(raw_words))))
                return pred_entities, 0, 0, 0
        pred_labels = [pred_text]

    pred_dicts = {}
    for p_label in pred_labels:
        p_label = p_label.strip()
        if "：" in p_label:
            ent_type = p_label.split('：')[0].strip()
            ent_spans = p_label.split('：')[1]
        elif ":" in p_label:
            ent_type = p_label.split(':')[0].strip()
            ent_spans = p_label.split(':')[1]
        elif p_label in label2desc.values():
            ent_type = p_label
            ent_spans = raw_text.strip()
        else:
            format_negative += 1
            continue
        for t in list(label2desc.values()):
            if t.lower() in ent_type and t.lower() in label_list:
                ent_type = label2desc.inverse[t]
                if '，' in ent_spans:
                    pred_dicts[ent_type] = ent_spans.split('，')
                elif '、' in ent_spans:
                    pred_dicts[ent_type] = ent_spans.split('、')
                else:
                    pred_dicts[ent_type] = [ent_spans]
                break
        if ent_type not in list(label2desc.keys()):
            span_negative += 1
            continue

    # split event spans with same event type
    for ent_type in pred_dicts:
        ent_spans = pred_dicts[ent_type]
        for ent_span in ent_spans:
            ent_span = ent_span.strip()
            if len(ent_span) == 0:
                span_negative += 1
                continue
            if ent_span == '无':
                continue
            if ent_span[-1] == '。' or ent_span[-1] == '；':
                ent_span = ent_span[:-1]
            if ent_span not in raw_text:
                span_negative += 1
                continue
            try:
                starts = [s_idx for s_idx, word in enumerate(raw_words) if word == ent_span[0]]
                assert len(starts) > 0
                for start in starts:
                    end = start + len(ent_span)
                    if end - 1 >= len(raw_words) or raw_words[end - 1] != ent_span[-1]:
                        span_negative += 1
                        continue
                    pred_entities.append((ent_type.lower(), (start, end)))
            except (ValueError, AssertionError):
                span_negative += 1

    if format_negative > 0 or span_negative > 0:
        sent_invalid = 1
    else:
        sent_invalid = 0

    return pred_entities, sent_invalid, format_negative, span_negative


def format_en_cot_entity(pred_text, raw_words, gold_types):
    raw_words = [word.lower() for word in raw_words]
    raw_text = ' '.join(raw_words).lower()
    pred_text = pred_text.lower().strip()
    pred_entities = []
    format_negative = 0
    span_negative = 0

    if pred_text == 'none' or pred_text == 'none.' or pred_text == 'no' or pred_text == 'no.':
        return pred_entities, 0, 0, 0

    # locate the entity
    if "therefore, " not in pred_text:
        return pred_entities, 1, 1, 0
    pred_labels = pred_text.split('therefore, ')[1:]
    for p_label in pred_labels:
        p_label = p_label.split(';')[0].strip()
        if " is a " in p_label:
            p_label = p_label.split(' is a ')
            ent_spans = [p_label[0]]
        elif " is an " in p_label:
            p_label = p_label.split(' is an ')
            ent_spans = [p_label[0]]
        elif ' are ' in p_label:
            p_label = p_label.split(' are ')
            if ' and ' in p_label[0]:
                ent_spans = p_label[0].split(' and ')
            elif ', ' in p_label[0]:
                ent_spans = p_label[0].split(', ')
            else:
                format_negative += 1
                continue
        else:
            format_negative += 1
            continue
        try:
            # split entity spans and entity types
            ent_type = p_label[1].strip().split(' ')[0]
            for gold_type in gold_types:
                if gold_type.lower() in ent_type:
                    ent_type = gold_type
                    break
            assert ent_type in gold_types
            ent_type = label2desc.inverse[ent_type]
            for ent_span in ent_spans:
                ent_span = ent_span.strip()
                # split entity spans with same entity type
                if len(ent_span) == 0 or ent_span not in raw_text:
                    span_negative += 1
                    continue
                if ent_span == 'none':
                    continue
                if ent_span[-1] == '.' or ent_span[-1] == ';':
                    ent_span = ent_span[:-1]
                if '-' in ent_span and ' - ' not in ent_span and ' - ' in raw_text:
                    ent_span = ' - '.join(ent_span.split('-'))
                if '(' in ent_span and ')' in ent_span and ' ( ' in raw_text:
                    ent_span = ' ( '.join(ent_span.split('('))
                    ent_span = ' ) '.join(ent_span.split(')'))
                ent_span = ent_span.split()
                if len(ent_span) == 0:
                    span_negative += 1
                    continue
                starts = [s_idx for s_idx, word in enumerate(raw_words) if word == ent_span[0]]
                if len(starts) == 0:
                    span_negative += 1
                    continue
                for start in starts:
                    end = start + len(ent_span)
                    if end - 1 >= len(raw_words) or raw_words[end - 1] != ent_span[-1]:
                        span_negative += 1
                        continue
                    pred_entities.append((ent_type.lower(), (start, end)))
        except (ValueError, AssertionError):
            span_negative += 1

    if format_negative > 0 or span_negative > 0:
        sent_invalid = 1
    else:
        sent_invalid = 0

    return pred_entities, sent_invalid, format_negative, span_negative


def format_cn_cot_entity(pred_text, raw_words, gold_types):
    raw_text = ''.join(raw_words)
    pred_text = pred_text.lower().strip()
    pred_entities = []
    format_negative = 0
    span_negative = 0

    if pred_text == '无' or pred_text == '无。':
        return pred_entities, 0, 0, 0

    # locate the entity
    if "因此，" not in pred_text:
        return pred_entities, 1, 1, 0
    pred_labels = pred_text.split('因此，')[1:]
    for p_label in pred_labels:
        p_label = p_label.split('；')[0].strip()
        if "是" in p_label:
            p_label = p_label.split('是')
        elif "是一个" in p_label:
            p_label = p_label.split('是一个')
        else:
            format_negative += 1
            continue
        if '，' in p_label[0]:
            ent_spans = p_label[0].split('，')
        elif '、' in p_label[0]:
            ent_spans = p_label[0].split('、')
        else:
            ent_spans = [p_label[0]]
        try:
            # split entity spans and entity types
            ent_type = p_label[1].strip().split('实体')[0]
            for gold_type in gold_types:
                if gold_type.lower() in ent_type:
                    ent_type = gold_type
                    break
            assert ent_type in gold_types
            ent_type = label2desc.inverse[ent_type]
            for ent_span in ent_spans:
                ent_span = ent_span.strip()
                # split entity spans with same entity type
                if len(ent_span) == 0 or ent_span not in raw_text:
                    span_negative += 1
                    continue
                if ent_span == '无':
                    continue
                if ent_span[-1] == '。' or ent_span[-1] == '；':
                    ent_span = ent_span[:-1]
                starts = [s_idx for s_idx, word in enumerate(raw_words) if word == ent_span[0]]
                if len(starts) == 0:
                    span_negative += 1
                    continue
                for start in starts:
                    end = start + len(ent_span)
                    if end - 1 >= len(raw_words) or raw_words[end - 1] != ent_span[-1]:
                        span_negative += 1
                        continue
                    pred_entities.append((ent_type.lower(), (start, end)))
        except (ValueError, AssertionError):
            span_negative += 1

    if format_negative > 0 or span_negative > 0:
        sent_invalid = 1
    else:
        sent_invalid = 0

    return pred_entities, sent_invalid, format_negative, span_negative