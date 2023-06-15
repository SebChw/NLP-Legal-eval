from typing import Dict, List, Optional, Union, Any

class Dataset(dict):
    @property
    def features(self): ...
    def remove_columns(self, columns: List[str]): ...
    def map(
        self,
        func,
        remove_columns: List[str],
        batch_size :Optional[Any] = None,
        batched :Optional[Any] = None,
        fn_kwargs :Optional[Any] = None,
        load_from_cache_file :Optional[Any] = None,
    ): ...
    def set_format(self, format: str, columns: List[str]): ...
    def select(
        self,
        indices,
    ): ...

class ClassLabel:
    def __init__(self, names: List[str]): ...

class DatasetDict(dict):
    def __init__(self, splits: Dict[str, Dataset]): ...
    def remove_columns(self, columns: List[str]): ...
    def map(
        self, func, remove_columns: List[str], batched :Optional[Any] = None, fn_kwargs :Optional[Any] = None, load_from_cache_file :Optional[Any] = None
    ): ...
    def cast(self, features: Features): ...
    def set_format(self, format: str, columns: List[str]): ...

class Features:
    def __init__(self, features: Dict[str, Union[Sequence, Value]]): ...

class Sequence:
    def __init__(self, class_label: Union[ClassLabel, Value]): ...

class Value:
    def __init__(self, dtype: str): ...

def load_dataset(
    path: str,
    name: Optional[str] = None,
    data_dir: Optional[str] = None,
    data_files: Optional[List[str]] = None,
    cache_dir: Optional[str] = None,
    features: Optional[Features] = None,
    ignore_verifications="deprecated",
    keep_in_memory: Optional[bool] = None,
    save_infos: bool = False,
    use_auth_token: Optional[Union[bool, str]] = None,
    streaming: bool = False,
    num_proc: Optional[int] = None,
    storage_options: Optional[Dict] = None,
    **config_kwargs,
) -> DatasetDict: ...
