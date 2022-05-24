import numpy as np
from sklearn.metrics import f1_score

from typing import List, Union, NamedTuple, Tuple, Counter
from ee_data import EE_label2id, EE_label2id1, EE_label2id2, EE_id2label1, EE_id2label2, EE_id2label, NER_PAD, \
    _LABEL_RANK


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class ComputeMetricsForNER:  # training_args  `--label_names labels `
    def __call__(self, eval_pred) -> dict:
        predictions, labels = eval_pred

        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]
        labels[labels == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]

        '''NOTE: You need to finish the code of computing f1-score.

        '''
        pred_flatten = []
        label_flatten = []
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 0:
                    continue
                pred_flatten.append(predictions[i, j])
                label_flatten.append(labels[i, j])
        pred_flatten = np.array(pred_flatten, dtype=int)
        label_flatten = np.array(label_flatten, dtype=int)
        micro_f1 = f1_score(y_true=label_flatten, y_pred=pred_flatten, average='micro')  # 0.8444585709615314
        macro_f1 = f1_score(y_true=label_flatten, y_pred=pred_flatten, average='macro')  # 0.6597221115955032

        return {"f1": micro_f1}


class ComputeMetricsForNestedNER:  # training_args  `--label_names labels labels2`
    def __call__(self, eval_pred) -> dict:
        predictions, (labels1, labels2) = eval_pred

        # -100 ==> [PAD]
        predictions[predictions == -100] = EE_label2id[NER_PAD]  # [batch, seq_len, 2]
        labels1[labels1 == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]
        labels2[labels2 == -100] = EE_label2id[NER_PAD]  # [batch, seq_len]

        '''NOTE: You need to finish the code of computing f1-score.

        '''
        pred_flatten = []
        label_flatten = []
        for k in range(2):
            for i in range(labels1.shape[0]):
                for j in range(labels1.shape[1]):
                    if labels1[i, j] == 0:
                        continue
                    pred_flatten.append(predictions[i, j, k])
                    if k == 0:
                        label_flatten.append(labels1[i, j])
                    else:
                        label_flatten.append(labels2[i, j])

        pred_flatten = np.array(pred_flatten, dtype=int)
        label_flatten = np.array(label_flatten, dtype=int)
        micro_f1 = f1_score(y_true=label_flatten, y_pred=pred_flatten, average='micro')  # 0.9074121835155654
        macro_f1 = f1_score(y_true=label_flatten, y_pred=pred_flatten, average='macro')  # 0.6799959800219424

        return {"f1": micro_f1}


def extract_entities(batch_labels_or_preds: np.ndarray, for_nested_ner: bool = False, first_labels: bool = True) -> \
List[List[tuple]]:
    """
    本评测任务采用严格 Micro-F1作为主评测指标, 即要求预测出的 实体的起始、结束下标，实体类型精准匹配才算预测正确。

    Args:
        batch_labels_or_preds: The labels ids or predicted label ids.
        for_nested_ner:        Whether the input label ids is about Nested NER.
        first_labels:          Which kind of labels for NestNER.
    """
    batch_labels_or_preds[batch_labels_or_preds == -100] = EE_label2id1[NER_PAD]  # [batch, seq_len]

    if not for_nested_ner:
        id2label = EE_id2label
    else:
        id2label = EE_id2label1 if first_labels else EE_id2label2

    batch_entities = []  # List[List[(start_idx, end_idx, type)]]

    # '''NOTE: You need to finish this function of extracting entities for generating results and computing metrics.

    # '''
    for i in range(batch_labels_or_preds.shape[0]):
        entities = []

        j = 0
        while j < batch_labels_or_preds.shape[1]:
            label_id = batch_labels_or_preds[i, j]
            label_str = id2label[label_id]
            label_sp = label_str.split('-')

            if label_sp[0] == 'B':
                label_type = label_sp[1]
                for k in range(j + 1, batch_labels_or_preds.shape[1]):
                    label_k = id2label[batch_labels_or_preds[i, k]]
                    if label_k != f'I-{label_type}':
                        entity = (j, k - 1, label_type)
                        entities.append(entity)
                        j = k + 1
                        break
            else:
                j += 1

        batch_entities.append(entities)

    return batch_entities


if __name__ == '__main__':

    # Test for ComputeMetricsForNER
    predictions = np.load('../test_files/predictions.npy')
    labels = np.load('../test_files/labels.npy')

    metrics = ComputeMetricsForNER()(EvalPrediction(predictions, labels))

    if abs(metrics['f1'] - 0.606179116) < 1e-5:
        print('You passed the test for ComputeMetricsForNER.')
    else:
        print('The result of ComputeMetricsForNER is not right.')

    # Test for ComputeMetricsForNestedNER
    predictions = np.load('../test_files/predictions_nested.npy')
    labels1 = np.load('../test_files/labels1_nested.npy')
    labels2 = np.load('../test_files/labels2_nested.npy')

    metrics = ComputeMetricsForNestedNER()(EvalPrediction(predictions, (labels1, labels2)))

    if abs(metrics['f1'] - 0.60333644) < 1e-5:
        print('You passed the test for ComputeMetricsForNestedNER.')
    else:
        print('The result of ComputeMetricsForNestedNER is not right.')
