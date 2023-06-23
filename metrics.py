from collections import Counter

from fastNLP import Metric
from fastNLP.core.metrics.utils import _compute_f_pre_rec

class NERSpanMetrics(Metric):
    def __init__(self, f_type='micro', beta=1, backend='auto', aggregate_when_get_metric: bool = None):
        super().__init__(backend, aggregate_when_get_metric)
        self.sent_invalid = 0
        self.format_negative = 0
        self.span_negative = 0
        self.false_positive = 0
        self.true_positive = 0
        self.negative = 0

        self._tp = Counter()
        self._fp = Counter()
        self._fn = Counter()
        self.f_type = f_type
        self.beta = beta
        self.beta_square = self.beta ** 2

    def update(self, pred, target, invalid_sent=0, format_negative=0, span_negative=0,
               false_positive=0, true_positive=0, negative=0):
        batch_size = len(pred)
        for i in range(batch_size):
            for span in pred[i]:
                if span in target[i]:
                    self._tp[span[0]] += 1
                    target[i].remove(span)
                else:
                    self._fp[span[0]] += 1
            for span in target[i]:
                self._fn[span[0]] += 1

        self.sent_invalid += invalid_sent
        self.format_negative += format_negative
        self.span_negative += span_negative
        self.false_positive += false_positive
        self.true_positive += true_positive
        self.negative += negative

    def reset(self):
        """
        重置所有元素
        """
        self._tp.clear()
        self._fp.clear()
        self._fn.clear()

    def get_metric(self) -> dict:
        """
        :meth:`get_metric` 函数将根据 :meth:`update` 函数累计的评价指标统计量来计算最终的评价结果。
        """
        evaluate_result = {}

        # 通过 all_gather_object 将各个卡上的结果收集过来，并加和。
        ls = self.all_gather_object([self._tp, self._fp, self._fn])
        tps, fps, fns = zip(*ls)
        _tp, _fp, _fn = Counter(), Counter(), Counter()
        for c, cs in zip([_tp, _fp, _fn], [tps, fps, fns]):
            for _c in cs:
                c.update(_c)

        if self.f_type == 'macro':
            tags = set(_fn.keys())
            tags.update(_fp.keys())
            tags.update(_tp.keys())
            f_sum = 0
            pre_sum = 0
            rec_sum = 0
            for tag in tags:
                tp = _tp[tag]
                fn = _fn[tag]
                fp = _fp[tag]
                if tp == fn == fp == 0:
                    continue

                f, pre, rec = _compute_f_pre_rec(self.beta_square, tp, fn, fp)
                f_sum += f
                pre_sum += pre
                rec_sum += rec
                if not self.only_gross and tag != '':  # tag!=''防止无tag的情况
                    f_key = 'f-{}'.format(tag)
                    pre_key = 'pre-{}'.format(tag)
                    rec_key = 'rec-{}'.format(tag)
                    evaluate_result[f_key] = f
                    evaluate_result[pre_key] = pre
                    evaluate_result[rec_key] = rec

            if self.f_type == 'macro':
                evaluate_result['f'] = f_sum / len(tags)
                evaluate_result['pre'] = pre_sum / len(tags)
                evaluate_result['rec'] = rec_sum / len(tags)

        if self.f_type == 'micro':
            f, pre, rec = _compute_f_pre_rec(self.beta_square, sum(_tp.values()), sum(_fn.values()), sum(_fp.values()))
            evaluate_result['f'] = f
            evaluate_result['pre'] = pre
            evaluate_result['rec'] = rec

        for key, value in evaluate_result.items():
            evaluate_result[key] = round(value, 6)

        evaluate_result['sent_invalid'] = self.sent_invalid
        evaluate_result['format_negative'] = self.format_negative
        evaluate_result['span_negative'] = self.span_negative
        evaluate_result['false_positive'] = self.false_positive
        evaluate_result['true_positive'] = self.true_positive
        evaluate_result['negative'] = self.negative

        return evaluate_result


if __name__ == '__main__':
    print(1)