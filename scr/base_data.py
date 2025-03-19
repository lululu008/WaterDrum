import os
import datasets
import pandas as pd


def load_tofu_train_dataset(
    dataset_dir: str = None,
    hf_dataset_name: str = None,
    hf_dataset_split: str = None,
    is_wtm: bool = False,
    forget_ratio: float = 0.0,
    forget_calibration_ratio: float = 0.0,
    forget_dup_ratio: float = 0.0,
    train_dup_data_dir: str = None,
):
    if dataset_dir is not None:
        train_data = datasets.load_from_disk(dataset_dir)
    else:
        train_data = datasets.load_dataset(hf_dataset_name, hf_dataset_split)['train']
    forget_data, retain_data = split_dataset(train_data, forget_ratio)
    if forget_data is not None:
        # append the duplicated forget set to retain_data and train_data
        train_data, retain_data = duplicate_forget_data(train_data,
                                                        forget_data,
                                                        retain_data,
                                                        forget_dup_ratio,
                                                        train_dup_data_dir,
                                                        is_wtm,
                                                        )
        # calibrate the forget_set and append the remaining of the forget_set to the retain_data
        forget_data, retain_data = calibrate_forget_data(forget_data,
                                                         retain_data,
                                                         forget_ratio,
                                                         forget_calibration_ratio)

    return train_data, forget_data, retain_data


def load_arxiv_train_dataset(
    dataset_dir: str,
    hf_dataset_name: str = None,
    hf_dataset_split: str = None,
    is_wtm: bool = False,
    forget_ratio: float = 0.0,
    forget_calibration_ratio: float = 0.0,
    forget_dup_ratio: float = 0.0,
    train_dup_data_dir: str = None,
    **kwargs 
):
    if dataset_dir is not None:
        df = pd.read_pickle(dataset_dir)    # Abused name: assume dataset_dir is a pickle file
        if is_wtm and 'watermarked' in df.columns:
            text = df['watermarked'].tolist()
        else:
            text = df['Summary'].tolist()
        train_data = datasets.Dataset.from_dict({'text': text})
    else:
        train_data = datasets.load_dataset(hf_dataset_name, hf_dataset_split)["full"]

    forget_data, retain_data = split_dataset(train_data, forget_ratio)
    if forget_data is not None:
        train_data, retain_data = duplicate_forget_data(train_data,
                                                        forget_data,
                                                        retain_data,
                                                        forget_dup_ratio,
                                                        train_dup_data_dir,
                                                        is_wtm,
                                                        is_arxiv_data=True,
                                                        )
        forget_data, retain_data = calibrate_forget_data(forget_data,
                                                         retain_data,
                                                         forget_ratio,
                                                         forget_calibration_ratio)
    return train_data, forget_data, retain_data


def split_dataset(train_data, forget_ratio: float = None):
    if forget_ratio is None or forget_ratio == 0.0:
        print(f'[WARNING] Forget set is empty (forget_ratio = {forget_ratio})')
        forget_data = None
        retain_data = train_data
    else:
        assert 0.0 < forget_ratio <= 1.0
        total_num_rows = len(train_data)
        forget_num_rows = int(total_num_rows * forget_ratio)
        retain_num_rows = total_num_rows - forget_num_rows
        retain_data = train_data.select(range(0, retain_num_rows))
        forget_data = train_data.select(range(retain_num_rows, total_num_rows))

    return forget_data, retain_data


def duplicate_forget_data(
    train_data,
    forget_data,
    retain_data,
    forget_dup_ratio: float = 0.0,
    train_dup_data_dir: str = None,
    is_wtm: bool = False,
    is_arxiv_data: bool = False,
):
    total_num_rows = len(train_data)
    if forget_dup_ratio != 0:
        print('Duplicating forget set...')
        dup_num_rows = int(forget_dup_ratio * total_num_rows)

        if train_dup_data_dir is None:
            # If train_dup_data_dir is not provided, we would assume exact duplication of last forget_num_rows samples of the forget set
            print(f'Exactly duplicated based on the forget set')
            forget_num_rows = len(forget_data)
            dup_data = forget_data.select(range(forget_num_rows - dup_num_rows, forget_num_rows))
        
        else:
            # Otherwise, we will choose the duplicated data from the train_dup_data
            print(f'Loading duplicated train set from {train_dup_data_dir}')
            
            if is_arxiv_data:
                df = pd.read_pickle(train_dup_data_dir)
                if "watermarked" in df.columns:
                    text = df["watermarked"].to_list()
                else:
                    if "paraphrased" in train_dup_data_dir:
                        text = df["paraphrased_Summary"].tolist()
                    else:
                        text = df["Summary"].tolist()
                train_dup_data = datasets.Dataset.from_dict({'text': text})
                dup_data = train_dup_data.select(range(total_num_rows - dup_num_rows, total_num_rows))

            else:   # TOFU
                if train_dup_data_dir.endswith('.pkl'):
                    train_dup_data = pd.read_pickle(train_dup_data_dir)
                    # combine original question-answer pairs and alternatively watermarked answers
                    forget_num_rows = len(forget_data)
                    dup_data = forget_data.select(range(forget_num_rows - dup_num_rows, forget_num_rows))
                    dup_data = dup_data.add_column('answer_watermarked', train_dup_data['watermarked'][-dup_num_rows:])
                    dup_data = dup_data.to_dict()
                    dup_data['answer_split'] = dup_data['answer_watermarked']
                    dup_data.pop('answer_watermarked')
                    dup_data = datasets.Dataset.from_dict(dup_data)
                else:
                    train_dup_data = datasets.load_from_disk(train_dup_data_dir)
                    assert len(train_dup_data) == total_num_rows
                    # and take the forget_num_rows sample of the duplicated train set
                    dup_data = train_dup_data.select(range(total_num_rows - dup_num_rows, total_num_rows))
                    dup_data = dup_data.to_dict()
                    all_keys = list(dup_data.keys())

                    if is_wtm:
                        answer_col = 'answer_split'
                        question_col = 'question'
                        dup_data[answer_col] = dup_data['answer_watermarked']
                    else:
                        answer_col = 'answer'
                        question_col = 'question'
                        if 'paraphrased_answer' in dup_data:
                            dup_data[answer_col] = dup_data['paraphrased_answer']
                        
                    for k in all_keys:
                        if k not in (answer_col, question_col):
                            dup_data.pop(k)
                    
                    dup_data = datasets.Dataset.from_dict(dup_data)
                    diff_cols = set(dup_data.column_names).difference(set(forget_data.column_names))
                    assert len(diff_cols) == 0, f'These columns are different between duplicated and forget subset: {diff_cols}'

        # add duplicated samples to retain set and train set
        train_data = datasets.concatenate_datasets([train_data, dup_data])
        retain_data = datasets.concatenate_datasets([retain_data, dup_data])
        print('num_duplicated_rows:', len(dup_data))

    return train_data, retain_data


def calibrate_forget_data(
    forget_data,
    retain_data,
    forget_ratio: float = 0.0,
    forget_calibration_ratio: float = 0.0,
):
    if forget_calibration_ratio != forget_ratio:
        print('Calibrating forget set...')
        forget_num_rows = int(forget_calibration_ratio / forget_ratio * len(forget_data))
        calibrated_forget_data = forget_data.select(range(forget_num_rows))
        added_retain_data = forget_data.select(range(forget_num_rows, len(forget_data)))
        forget_data = calibrated_forget_data
        retain_data = datasets.concatenate_datasets([retain_data, added_retain_data])

    return forget_data, retain_data
