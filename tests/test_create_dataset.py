import os
import pytest

from create_dataset import unique_id, check_basedir


def test_unique_id():
    uniq_gen = unique_id()
    id_1 = next(uniq_gen)
    id_2 = next(uniq_gen)
    assert id_1 != id_2


def test_folder_does_not_exists_check_basedir():
    with pytest.raises(Exception):
        check_basedir("NONE")


def test_folder_exists_check_basedir():
    check_basedir(os.getcwd())
