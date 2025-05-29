import pytest
import platform
import os

ON_MACOS = platform.system() == "Darwin"
IN_GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.fixture(scope="session")
def prep_temp_dir(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_msic_utils_temp")
    return fn


@pytest.mark.skipif(
    IN_GITHUB_ACTIONS,
    reason="Test is flaky on the CI runners as it relies on spawning multiple threads. ",
)
def test_filelocking(prep_temp_dir):
    from modelforge.utils.misc import lock_file, unlock_file, check_file_lock

    filepath = str(prep_temp_dir) + "/test.txt"

    import threading
    import time

    class thread(threading.Thread):
        def __init__(self, thread_name, thread_id, filepath):
            threading.Thread.__init__(self)
            self.thread_id = thread_id
            self.name = thread_name
            self.filepath = filepath
            self.did_I_lock_it = None

        def run(self):

            if self.name == "lock_file_here":
                with open(self.filepath, "w") as f:
                    if not check_file_lock(f):
                        lock_file(f)
                        self.did_I_lock_it = True
                        time.sleep(3)
                        # unlock_file(f)
            else:
                with open(self.filepath, "w") as f:
                    if check_file_lock(f):

                        self.did_I_lock_it = False

    # the first thread should lock the file and set "did_I_lock_it" to True
    thread1 = thread("lock_file_here", "Thread-1", filepath)
    # the second thread should check if locked, and set "did_I_lock_it" to False
    # the second thread should also set "status" to True, because it waits
    # for the first thread to unlock the file
    thread2 = thread("encounter_locked_file", "Thread-2", filepath)

    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()

    assert thread1.did_I_lock_it == True

    assert thread2.did_I_lock_it == False


def test_unzip_file(prep_temp_dir):
    from modelforge.utils.misc import ungzip_file

    from importlib import resources
    from modelforge.tests import data

    file_input_path = str(resources.files(data)._paths[0])

    ungzip_file(file_input_path, "test_file.txt.gz", prep_temp_dir)
    assert os.path.isfile(str(prep_temp_dir) + "/test_file.txt")

    with open(str(prep_temp_dir) + "/test_file.txt", "r") as f:
        assert f.read().strip() == "12345"
